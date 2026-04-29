# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset catalog - loading and script-based provisioning."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """A single dataset declared in the catalog."""

    name: str
    script: str  # path to preparation script (relative to repo root)
    size_tier: str  # tiny | small | medium | large
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_path(self) -> Path:
        """Return the conventional dataset directory: ``<name>``."""
        return Path(self.name)


@dataclass(frozen=True)
class DatasetCatalog:
    """Parsed representation of ``benchmark_catalog.yaml``."""

    version: int
    datasets: dict[str, DatasetEntry]  # name -> entry

    # -- querying ----------------------------------------------------------

    def all_entries(self) -> list[DatasetEntry]:
        """Return every dataset entry."""
        return list(self.datasets.values())

    def filter(
        self,
        *,
        size_tiers: list[str] | None = None,
        names: set[str] | None = None,
    ) -> list[DatasetEntry]:
        """Return entries matching **all** supplied filters."""
        results: list[DatasetEntry] = []
        for entry in self.datasets.values():
            if size_tiers and entry.size_tier not in size_tiers:
                continue
            if names and entry.name not in names:
                continue
            results.append(entry)
        return results

    def get(self, name: str) -> DatasetEntry:
        """Look up a single dataset by name."""
        if name not in self.datasets:
            msg = f"Dataset '{name}' not found in catalog."
            raise KeyError(msg)
        return self.datasets[name]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> DatasetCatalog:
    """Parse ``benchmark_catalog.yaml`` into a :class:`DatasetCatalog`."""
    with Path(path).open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    version = raw.get("version", 1)
    datasets: dict[str, DatasetEntry] = {}
    known_keys = {f.name for f in DatasetEntry.__dataclass_fields__.values()}
    for entry_raw in raw.get("datasets", []):
        core = {k: v for k, v in entry_raw.items() if k in known_keys}
        extra = {k: v for k, v in entry_raw.items() if k not in known_keys}
        entry = DatasetEntry(extra=extra, **core)
        datasets[entry.name] = entry
    return DatasetCatalog(version=version, datasets=datasets)


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def _resolve_script_path(script: str) -> Path:
    """Resolve a script path relative to the repository root.

    The *script* field in the catalog is relative to the repo root.
    We walk up from the ``catalog.py`` source file to find ``library/``.
    """
    # catalog.py is at library/src/getitune/benchmark/catalog.py
    # repo root is 4 levels up (src/getitune/benchmark/catalog.py -> library/)
    library_root = Path(__file__).resolve().parents[3]
    return library_root / script


def _run_script(script_path: Path, data_root: Path, name: str) -> None:
    """Execute a dataset preparation script and stream its output in real time.

    The child's stdout/stderr are merged and forwarded line-by-line to the
    logger so that long-running download/extract steps are visible to the
    user.

    The call blocks until the child exits; a non-zero exit code raises
    :class:`RuntimeError`.
    """
    logger.info("Running preparation script: %s (dataset=%s)", script_path, name)

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--output-dir",
        str(data_root),
        "--name",
        name,
    ]

    # Merge stderr into stdout so log ordering matches the child's own
    # write order (and so a single reader loop is sufficient).
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    assert proc.stdout is not None  # noqa: S101 - guaranteed by PIPE above
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if line:
                logger.info("  [%s] %s", name, line)
    finally:
        # Always wait for the child so we never return before it finishes.
        returncode = proc.wait()

    if returncode != 0:
        msg = f"Preparation script for dataset '{name}' failed with exit code {returncode}.\nScript: {script_path}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def provision_dataset(entry: DatasetEntry, data_root: Path) -> Path:
    """Ensure a single dataset is prepared and ready.

    A ``.ready`` sentinel file is written inside the dataset directory once
    the preparation script finishes successfully. On subsequent runs the
    sentinel is what we check — a stale, half-populated directory left
    behind by a crashed prep run is therefore treated as "not ready" and
    the script is re-executed.

    Returns the path to the prepared dataset directory.
    """
    dataset_dir = data_root / entry.relative_path
    ready_marker = dataset_dir / ".ready"

    if ready_marker.exists():
        logger.info("Dataset '%s' already exists, skipping.", entry.name)
        return dataset_dir

    if dataset_dir.exists():
        logger.warning(
            "Dataset '%s' directory exists but is missing the readiness marker; re-running prep.",
            entry.name,
        )

    script_path = _resolve_script_path(entry.script)

    if not script_path.exists():
        msg = f"Preparation script not found for dataset '{entry.name}': {script_path}"
        raise FileNotFoundError(msg)

    _run_script(script_path, data_root, entry.name)

    if not dataset_dir.exists():
        msg = f"Preparation script for '{entry.name}' did not create expected directory: {dataset_dir}"
        raise RuntimeError(msg)

    # Write the readiness sentinel only after a clean run so an interrupted
    # prep is never mistaken for a successful one on a later resume.
    ready_marker.touch()

    return dataset_dir


def provision_datasets(
    catalog: DatasetCatalog,
    data_root: Path,
    *,
    entries: list[DatasetEntry] | None = None,
) -> dict[str, Path]:
    """Run preparation scripts for all datasets (or a filtered subset).

    Returns a mapping ``{dataset_name: prepared_path}``.
    """
    targets = entries if entries is not None else catalog.all_entries()
    result: dict[str, Path] = {}
    for entry in targets:
        result[entry.name] = provision_dataset(entry, data_root)
    return result
