#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'bccd' benchmark dataset.

Downloads the BCCD (Blood Cell Count and Detection) dataset from GitHub and
exports it in the experimental Datumaro dataset format for detection benchmarks.

BCCD ships in standard Pascal VOC layout (``JPEGImages/``, ``Annotations/``,
``ImageSets/Main/``) but does **not** include a ``labelmap.txt``. Its custom
labels (RBC, WBC, Platelets) are not part of the default Pascal VOC label set,
so without a labelmap the VOC reader would drop every annotation. We therefore
write a ``labelmap.txt`` with BCCD's classes before loading, which lets the
experimental Datumaro VOC reader (``load_voc_dataset``) resolve the labels
directly and return ``VocSample`` instances. The official ``train`` / ``val`` /
``test`` split files are mapped to the TRAINING / VALIDATION / TESTING subsets.

BCCD contains microscopy images of blood smears annotated with three cell
types (RBC, WBC, Platelets). It contains no people and no other
privacy-sensitive content, making it a safe public detection benchmark.

Source: https://github.com/Shenggan/BCCD_Dataset
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from datumaro.experimental.data_formats.voc.io import load_voc_dataset
from datumaro.experimental.export_import import export_dataset

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

    from datumaro.experimental import Dataset

_COMMIT = "d272fb14cdff6e473fafeeeba32aba5f560e9e43"
_URL = f"https://github.com/Shenggan/BCCD_Dataset/archive/{_COMMIT}.zip"

# Class names are fixed by the dataset. The order defines the label indices, so
# it is kept stable here and written verbatim into the VOC ``labelmap.txt``.
_LABEL_NAMES = ("Platelets", "RBC", "WBC")

# Canonical split files mapped to TRAINING / VALIDATION / TESTING. BCCD also
# ships a ``trainval.txt`` (train plus val); the VOC reader maps it to TRAINING
# too, which would duplicate every train/val image. We drop it before loading.
_CANONICAL_SPLITS = ("train", "val", "test")

# Arbitrary but distinct colours for the VOC labelmap (only the name is used by
# the detection reader; colours matter only for segmentation masks).
_LABEL_COLORS = (
    (220, 20, 60),
    (0, 128, 0),
    (0, 0, 255),
)


def _write_labelmap(extracted_root: Path) -> None:
    """Write a Pascal VOC ``labelmap.txt`` so the reader resolves BCCD's labels.

    The VOC labelmap format is ``name:r,g,b:parts:actions``; only the name is
    used for detection. ``background`` is intentionally omitted so the label
    indices match :data:`_LABEL_NAMES` exactly.
    """
    lines = [f"{name}:{r},{g},{b}::" for name, (r, g, b) in zip(_LABEL_NAMES, _LABEL_COLORS, strict=True)]
    (extracted_root / "labelmap.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune_redundant_splits(extracted_root: Path) -> None:
    """Remove ImageSets/Main split files other than train/val/test.

    BCCD's ``trainval.txt`` overlaps ``train``/``val`` and the VOC reader maps it
    to TRAINING, which would otherwise duplicate those images in the dataset.
    """
    splits_dir = extracted_root / "ImageSets" / "Main"
    if not splits_dir.is_dir():
        return
    for txt_file in splits_dir.glob("*.txt"):
        if txt_file.stem not in _CANONICAL_SPLITS:
            txt_file.unlink()


def _load_bccd_dataset(extracted_root: Path) -> Dataset:
    """Load BCCD with the experimental Datumaro VOC reader.

    A ``labelmap.txt`` is written first so the reader resolves BCCD's custom
    labels instead of falling back to the default VOC label set, and the
    redundant ``trainval`` split is removed so images are not duplicated.
    """
    _write_labelmap(extracted_root)
    _prune_redundant_splits(extracted_root)
    dataset = load_voc_dataset(root_dir=str(extracted_root))

    loaded_labels = set(dataset.label_categories.labels) if dataset.label_categories is not None else set()
    if not set(_LABEL_NAMES).issubset(loaded_labels):
        msg = f"VOC reader did not resolve BCCD labels; got {sorted(loaded_labels)}"
        raise RuntimeError(msg)
    return dataset


def main() -> None:
    """Download BCCD, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the bccd benchmark dataset.")

    archive = download(_URL, dest_dir=args.archive_dir, filename=f"{args.name}.zip")

    # Extract into a temporary staging directory; ``args.dest`` will hold the
    # final Datumaro dataset.
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(archive, staging)
    archive.unlink(missing_ok=True)

    # The GitHub archive extracts into a single top-level
    # ``BCCD_Dataset-<commit>/`` folder; the data lives in its ``BCCD/`` subdir,
    # which follows the standard Pascal VOC directory layout.
    repo_root = next(p for p in staging.iterdir() if p.is_dir())
    extracted_root = repo_root / "BCCD"

    print("Loading BCCD with the experimental Datumaro VOC reader ...")
    dataset = _load_bccd_dataset(extracted_root)
    print(f"  Dataset length: {len(dataset)}")

    # ``export_dataset`` requires that the output path does NOT exist yet,
    # so remove any leftover from a previous run and let it create the dir.
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Clean up extracted archive contents.
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
