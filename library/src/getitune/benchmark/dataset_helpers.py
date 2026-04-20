# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for benchmark dataset preparation scripts.

Every preparation script in ``scripts/benchmark_datasets/`` follows the same
pattern: parse CLI args, download an archive, extract it, optionally transform
the data, and clean up.  This module provides reusable building blocks so that
individual scripts only need to define the dataset-specific logic.

Typical usage in a preparation script::

    #!/usr/bin/env python3
    from getitune.benchmark.dataset_helpers import (
        DatasetArgs,
        download,
        extract_archive,
        parse_args,
    )

    def main() -> None:
        args = parse_args(description="Prepare the pothole_tiny dataset.")

        archive = download(
            url="https://storage.geti.intel.com/test-data/pothole_tiny.tar.gz",
            dest_dir=args.archive_dir,
            filename="pothole_tiny.tar.gz",
        )

        extract_archive(archive, args.dest)

        # (optional) dataset-specific adjustments here …

        archive.unlink(missing_ok=True)
        print(f"Dataset '{args.name}' ready at {args.dest}")

    if __name__ == "__main__":
        main()
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetArgs:
    """Parsed CLI arguments common to all preparation scripts."""

    output_dir: Path
    """Root directory for dataset storage (``--output-dir``)."""

    name: str
    """Dataset name — determines the sub-directory (``--name``)."""

    @property
    def dest(self) -> Path:
        """Final dataset directory: ``<output_dir>/<name>``."""
        return self.output_dir / self.name

    @property
    def archive_dir(self) -> Path:
        """Temporary directory for downloaded archives: ``<output_dir>/.archives``."""
        return self.output_dir / ".archives"


def parse_args(*, description: str = "Prepare a benchmark dataset.") -> DatasetArgs:
    """Parse the standard ``--output-dir`` / ``--name`` CLI arguments.

    This should be the first call in every preparation script's ``main()``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name (determines sub-directory).",
    )
    ns = parser.parse_args()
    return DatasetArgs(output_dir=ns.output_dir, name=ns.name)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 1 << 20  # 1 MiB


def download(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """Download *url* into *dest_dir* and return the local file path.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest_dir:
        Directory to save the file in (created if missing).
    filename:
        Local file name.  Defaults to the last path segment of *url*.
    """
    if filename is None:
        filename = url.rsplit("/", 1)[-1]

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def extract_archive(archive: Path, dest: Path, *, clean_dest: bool = True) -> Path:
    """Extract a ``.tar.gz``, ``.tar``, or ``.zip`` archive into *dest*.

    Parameters
    ----------
    archive:
        Path to the archive file.
    dest:
        Directory to extract into (created if missing).
    clean_dest:
        If ``True`` (default) and *dest* already exists, it is removed
        before extraction so the result is always a clean copy.

    Returns:
    -------
    Path
        The *dest* directory.
    """
    if clean_dest and dest.exists():
        shutil.rmtree(dest)
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

    return dest
