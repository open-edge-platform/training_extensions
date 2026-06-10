#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'bccd' benchmark dataset.

Downloads the BCCD (Blood Cell Count and Detection) dataset from GitHub and
exports it in the experimental Datumaro dataset format for detection benchmarks.

BCCD ships in standard Pascal VOC layout (``JPEGImages/``, ``Annotations/``,
``ImageSets/Main/``), so we first try to load it directly with the experimental
Datumaro VOC reader (``load_voc_dataset``). BCCD does not, however, ship a
``labelmap.txt`` and its custom labels (RBC, WBC, Platelets) are not part of the
default Pascal VOC label set, so the auto-detected categories drop every
annotation. When that happens we fall back to building the dataset by hand using
``VocSample`` (xyxy bboxes) with the correct ``VocCategories``. Either path maps
the official ``train`` / ``val`` / ``test`` split files to the
TRAINING / VALIDATION / TESTING subsets.

BCCD contains microscopy images of blood smears annotated with three cell
types (RBC, WBC, Platelets). It contains no people and no other
privacy-sensitive content, making it a safe public detection benchmark.

Source: https://github.com/Shenggan/BCCD_Dataset
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.categories import MaskCategories
from datumaro.experimental.data_formats.voc.io import load_voc_dataset
from datumaro.experimental.data_formats.voc.sample import VocCategories, VocSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT = "d272fb14cdff6e473fafeeeba32aba5f560e9e43"
_URL = f"https://github.com/Shenggan/BCCD_Dataset/archive/{_COMMIT}.zip"

# BCCD ships official Pascal VOC split files under ``ImageSets/Main/``.
# Map each split directly to a Datumaro subset for a deterministic,
# reproducible assignment across runs.
_SPLIT_TO_SUBSET = {
    "train": Subset.TRAINING,
    "val": Subset.VALIDATION,
    "test": Subset.TESTING,
}

# Class names are fixed by the dataset. Listing them explicitly (sorted) keeps
# the label index mapping stable regardless of annotation parsing order.
_LABEL_NAMES = ("Platelets", "RBC", "WBC")


def _read_split(split_file: Path) -> list[str]:
    """Return the (sorted) image stems listed in a VOC split file."""
    if not split_file.exists():
        return []
    stems = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    return sorted(stems)


def _parse_voc_annotation(
    xml_path: Path,
    label_to_idx: dict[str, int],
) -> tuple[int, int, list[list[float]], list[int]]:
    """Parse a Pascal VOC XML file into image size, xyxy bboxes, and label indices.

    Args:
        xml_path: Path to the VOC annotation file.
        label_to_idx: Mapping from class name to contiguous label index.

    Returns:
        A tuple ``(width, height, bboxes, labels)`` where ``bboxes`` are in
        VOC ``[xmin, ymin, xmax, ymax]`` format and ``labels`` are integer class
        indices.
    """
    root = ET.parse(xml_path).getroot()  # noqa: S314 - trusted in-repo benchmark data

    size = root.find("size")
    width = int(size.findtext("width", default="0")) if size is not None else 0
    height = int(size.findtext("height", default="0")) if size is not None else 0

    bboxes: list[list[float]] = []
    labels: list[int] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None or name not in label_to_idx:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x_min = float(bndbox.findtext("xmin", default="0"))
        y_min = float(bndbox.findtext("ymin", default="0"))
        x_max = float(bndbox.findtext("xmax", default="0"))
        y_max = float(bndbox.findtext("ymax", default="0"))
        # VOC bboxes use absolute xyxy coordinates.
        bboxes.append([x_min, y_min, x_max, y_max])
        labels.append(label_to_idx[name])

    return width, height, bboxes, labels


def _build_voc_dataset(extracted_root: Path) -> Dataset:
    """Parse BCCD VOC annotations and build a Datumaro ``VocSample`` dataset.

    Used as a fallback when ``load_voc_dataset`` cannot resolve BCCD's custom
    labels. Bounding boxes are stored in VOC xyxy format and labels reference the
    explicit ``VocCategories`` built from :data:`_LABEL_NAMES`.
    """
    images_dir = extracted_root / "JPEGImages"
    ann_dir = extracted_root / "Annotations"
    splits_dir = extracted_root / "ImageSets" / "Main"

    label_to_idx = {name: idx for idx, name in enumerate(_LABEL_NAMES)}

    categories = VocCategories(labels=_LABEL_NAMES)
    # ``VocSample`` declares lazy mask fields whose schema requires
    # ``MaskCategories`` even though BCCD has no segmentation masks; build them
    # the same way the experimental VOC reader does so ``append`` validates.
    mask_categories = MaskCategories.generate(
        size=len(categories.labels),
        include_background=True,
        labels=list(categories.labels),
    )
    dataset: Dataset = Dataset(
        VocSample,
        categories={"labels": categories, "class_mask": mask_categories},
    )

    for split_name, subset in _SPLIT_TO_SUBSET.items():
        for stem in _read_split(splits_dir / f"{split_name}.txt"):
            xml_path = ann_dir / f"{stem}.xml"
            img_path = images_dir / f"{stem}.jpg"
            if not xml_path.exists() or not img_path.exists():
                continue

            width, height, bboxes_list, labels_list = _parse_voc_annotation(xml_path, label_to_idx)

            if bboxes_list:
                bboxes = np.asarray(bboxes_list, dtype=np.float32)
                labels = np.asarray(labels_list, dtype=np.uint32)
            else:
                bboxes = None
                labels = None

            dataset.append(
                VocSample(
                    image=LazyImage(img_path),
                    image_info=ImageInfo(width=width, height=height),
                    subset=subset,
                    bboxes=bboxes,
                    labels=labels,
                    difficult=None,
                    truncated=None,
                    occluded=None,
                    pose=None,
                    class_mask=None,
                    instance_mask=None,
                ),
            )

    return dataset


def _load_bccd_dataset(extracted_root: Path) -> Dataset:
    """Load BCCD, preferring the VOC reader and falling back to a manual build.

    First attempts the experimental Datumaro VOC reader. BCCD's labels are not
    part of the default VOC label set and it has no ``labelmap.txt``, so the
    reader may silently drop annotations; in that case we rebuild the dataset
    with explicit ``VocCategories``.
    """
    dataset = load_voc_dataset(root_dir=str(extracted_root))
    loaded_labels = set(dataset.label_categories.labels)
    if set(_LABEL_NAMES).issubset(loaded_labels):
        return dataset

    print("  VOC reader did not resolve BCCD labels; rebuilding with VocSample ...")
    return _build_voc_dataset(extracted_root)


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
