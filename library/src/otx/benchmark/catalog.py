# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset catalog - loading, downloading, and checksum verification."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = ".sha256"
_DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """A single dataset declared in the catalog."""

    name: str
    url: str
    sha256: str
    size_tier: str  # tiny | small | medium | large
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_path(self) -> Path:
        """Return the conventional extraction directory: ``<name>``."""
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
# Download & verification
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_DOWNLOAD_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _read_sentinel(dataset_dir: Path) -> str | None:
    sentinel = dataset_dir / SENTINEL_FILENAME
    if sentinel.exists():
        return sentinel.read_text().strip()
    return None


def _write_sentinel(dataset_dir: Path, sha: str) -> None:
    sentinel = dataset_dir / SENTINEL_FILENAME
    sentinel.write_text(sha + "\n")


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar."""
    import urllib.request

    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use urllib so we have no extra dependency beyond stdlib.
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct:3d}% ({downloaded}/{total})", end="", flush=True)
    print()  # newline after progress


def _extract(archive: Path, dest: Path) -> None:
    """Extract a ``.tar.gz`` or ``.zip`` archive into *dest*."""
    logger.info("Extracting %s -> %s", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)  # noqa: S202
    else:
        msg = f"Unsupported archive format: {archive}"
        raise ValueError(msg)


def provision_dataset(entry: DatasetEntry, data_root: Path) -> Path:
    """Ensure a single dataset is downloaded, verified, and extracted.

    Returns the path to the extracted dataset directory.
    """
    dataset_dir = data_root / entry.relative_path

    existing_sha = _read_sentinel(dataset_dir)
    if existing_sha == entry.sha256 and dataset_dir.exists():
        logger.info("Dataset '%s' is up-to-date (cache hit).", entry.name)
        return dataset_dir

    # Need to (re-)download
    if existing_sha and existing_sha != entry.sha256:
        logger.warning(
            "Dataset '%s' checksum mismatch (catalog changed?). Re-downloading.",
            entry.name,
        )

    archive_suffix = "".join(Path(entry.url).suffixes[-2:])  # e.g. ".tar.gz"
    archive_path = data_root / ".archives" / f"{entry.name}{archive_suffix}"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    _download(entry.url, archive_path)

    # Verify checksum
    actual_sha = _sha256_of_file(archive_path)
    if actual_sha != entry.sha256:
        msg = f"SHA-256 mismatch for dataset '{entry.name}': expected {entry.sha256}, got {actual_sha}"
        raise RuntimeError(msg)

    # Clean old extraction if present, then extract
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    _extract(archive_path, dataset_dir)

    _write_sentinel(dataset_dir, entry.sha256)

    # Clean up archive to save disk
    archive_path.unlink(missing_ok=True)

    return dataset_dir


def provision_datasets(
    catalog: DatasetCatalog,
    data_root: Path,
    *,
    entries: list[DatasetEntry] | None = None,
) -> dict[str, Path]:
    """Download and verify all datasets (or a filtered subset).

    Returns a mapping ``{dataset_name: extracted_path}``.
    """
    targets = entries if entries is not None else catalog.all_entries()
    result: dict[str, Path] = {}
    for entry in targets:
        result[entry.name] = provision_dataset(entry, data_root)
    return result
