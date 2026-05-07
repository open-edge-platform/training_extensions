#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Pascal VOC 2007 benchmark dataset.

Downloads Pascal VOC 2007 (~5k images, 20 object classes) with bounding box
and multi-label annotations. Exports in the experimental Datumaro dataset
format for detection and multi-label classification benchmarks.

Source: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_TRAINVAL_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

_VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def _parse_voc_annotation(xml_path: Path) -> dict:
    """Parse a single VOC XML annotation file."""
    tree = ET.parse(xml_path)  # noqa: S314
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    filename = root.find("filename").text

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        # Convert to COCO format: [x, y, w, h]
        objects.append(
            {
                "name": name,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "difficult": difficult,
            }
        )

    return {"filename": filename, "width": width, "height": height, "objects": objects}


def _build_dataset(voc_root: Path) -> Dataset:
    """Parse VOC 2007 annotations and build a Datumaro CocoSample dataset."""
    class_to_idx = {name: idx for idx, name in enumerate(_VOC_CLASSES)}

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=_VOC_CLASSES)},
    )

    # Read official train/val/test splits
    splits_dir = voc_root / "ImageSets" / "Main"
    ann_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"

    def _read_split_ids(split_name: str) -> set[str]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            return set()
        return {line.strip() for line in split_file.read_text().splitlines() if line.strip()}

    train_ids = _read_split_ids("train")
    val_ids = _read_split_ids("val")
    test_ids = _read_split_ids("test")

    all_ids = sorted(train_ids | val_ids | test_ids)

    for img_idx, img_id in enumerate(all_ids):
        xml_path = ann_dir / f"{img_id}.xml"
        if not xml_path.exists():
            continue

        ann = _parse_voc_annotation(xml_path)

        if img_id in train_ids:
            subset = Subset.TRAINING
        elif img_id in val_ids:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TESTING

        if ann["objects"]:
            bboxes = np.asarray([o["bbox"] for o in ann["objects"]], dtype=np.float32)
            labels = np.asarray(
                [class_to_idx[o["name"]] for o in ann["objects"] if o["name"] in class_to_idx],
                dtype=np.int64,
            )
            areas = bboxes[:, 2] * bboxes[:, 3]
            iscrowd = np.asarray(
                [o["difficult"] for o in ann["objects"]],
                dtype=np.int32,
            )
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            areas = np.zeros((0,), dtype=np.float32)
            iscrowd = np.zeros((0,), dtype=np.int32)

        dataset.append(
            CocoSample(
                image=LazyImage(images_dir / ann["filename"]),
                image_info=ImageInfo(width=ann["width"], height=ann["height"]),
                image_id=img_idx,
                subset=subset,
                bboxes=bboxes,
                labels=labels,
                polygons=np.empty((0,), dtype=object),
                areas=areas,
                iscrowd=iscrowd,
                caption_group_ids=None,
                captions=None,
                keypoints=None,
            ),
        )

    return dataset


def main() -> None:
    """Download Pascal VOC 2007, convert to Datumaro format, and save."""
    args = parse_args(description="Prepare the voc2007 benchmark dataset.")

    # Download train/val and test archives
    trainval_archive = download(_TRAINVAL_URL, dest_dir=args.archive_dir, filename=f"{args.name}_trainval.tar")
    test_archive = download(_TEST_URL, dest_dir=args.archive_dir, filename=f"{args.name}_test.tar")

    # Extract into staging
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(trainval_archive, staging, clean_dest=True)
    extract_archive(test_archive, staging, clean_dest=False)
    trainval_archive.unlink(missing_ok=True)
    test_archive.unlink(missing_ok=True)

    # VOC extracts into VOCdevkit/VOC2007/
    voc_root = staging / "VOCdevkit" / "VOC2007"

    print("Building Datumaro dataset from Pascal VOC 2007 ...")
    dataset = _build_dataset(voc_root)
    print(f"  Dataset length: {len(dataset)}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
