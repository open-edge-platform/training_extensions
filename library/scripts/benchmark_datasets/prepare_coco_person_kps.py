#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare a COCO person keypoints benchmark dataset.

Downloads COCO 2017 val split person keypoint annotations and the corresponding
images (person-only subset). Exports in the experimental Datumaro dataset format
for keypoint detection benchmarks.

Source: https://cocodataset.org/
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

# COCO 2017 val annotations (includes person_keypoints)
_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# COCO 2017 val images
_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

# COCO person keypoint skeleton definition
_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Only use the "person" category
_PERSON_CATEGORY = "person"

# Fraction of val images to use for validation vs test (deterministic split)
_VAL_FRACTION = 0.4


def _build_dataset(images_dir: Path, ann_file: Path) -> Dataset:
    """Parse COCO person keypoints annotations and build a Datumaro dataset."""
    with ann_file.open() as f:
        coco = json.load(f)

    # Build category mapping — we only keep "person"
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    person_cat_ids = {cid for cid, name in cat_id_to_name.items() if name == _PERSON_CATEGORY}

    if not person_cat_ids:
        msg = "No 'person' category found in annotations"
        raise ValueError(msg)

    label_names = (_PERSON_CATEGORY,)

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=label_names)},
    )

    # Index annotations by image_id, keeping only person annotations with keypoints
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in person_cat_ids:
            continue
        if "keypoints" not in ann or not ann["keypoints"]:
            continue
        # Only keep annotations that have at least some visible keypoints
        kps = ann["keypoints"]
        num_visible = sum(1 for i in range(2, len(kps), 3) if kps[i] > 0)
        if num_visible < 5:  # At least 5 visible keypoints
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Build image lookup
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Only include images that have person keypoint annotations
    valid_img_ids = sorted(anns_by_img.keys())

    # Deterministic split: first portion → validation, rest → test
    val_count = int(len(valid_img_ids) * _VAL_FRACTION)
    val_ids = set(valid_img_ids[:val_count])

    for _img_idx, img_id in enumerate(valid_img_ids):
        img_info = img_lookup[img_id]
        img_path = images_dir / img_info["file_name"]

        if not img_path.exists():
            continue

        subset = Subset.VALIDATION if img_id in val_ids else Subset.TESTING
        anns = anns_by_img[img_id]

        width = int(img_info["width"])
        height = int(img_info["height"])

        # Build arrays
        bboxes = np.asarray([a["bbox"] for a in anns], dtype=np.float32)
        labels = np.zeros((len(anns),), dtype=np.int64)  # all "person" = 0
        areas = np.asarray([a.get("area", 0.0) for a in anns], dtype=np.float32)
        iscrowd = np.asarray([a.get("iscrowd", 0) for a in anns], dtype=np.int32)

        # Keypoints: each annotation has a flat list [x1, y1, v1, x2, y2, v2, ...]
        # Convert to shape (num_instances, num_keypoints, 3)
        num_kps = len(_KEYPOINT_NAMES)
        keypoints = np.zeros((len(anns), num_kps, 3), dtype=np.float32)
        for i, ann in enumerate(anns):
            kps = ann["keypoints"]
            keypoints[i] = np.array(kps, dtype=np.float32).reshape(num_kps, 3)

        dataset.append(
            CocoSample(
                image=LazyImage(img_path),
                image_info=ImageInfo(width=width, height=height),
                image_id=img_id,
                subset=subset,
                bboxes=bboxes,
                labels=labels,
                polygons=np.empty((0,), dtype=object),
                areas=areas,
                iscrowd=iscrowd,
                caption_group_ids=None,
                captions=None,
                keypoints=keypoints,
            ),
        )

    return dataset


def main() -> None:
    """Download COCO person keypoints, convert to Datumaro format, and save."""
    args = parse_args(description="Prepare the coco_person_keypoints benchmark dataset.")

    # Download annotations and images
    ann_archive = download(_ANNOTATIONS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_annotations.zip")
    img_archive = download(_VAL_IMAGES_URL, dest_dir=args.archive_dir, filename=f"{args.name}_val2017.zip")

    # Extract
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(ann_archive, staging, clean_dest=True)
    extract_archive(img_archive, staging, clean_dest=False)
    ann_archive.unlink(missing_ok=True)
    img_archive.unlink(missing_ok=True)

    # Paths within extracted archive
    ann_file = staging / "annotations" / "person_keypoints_val2017.json"
    images_dir = staging / "val2017"

    print("Building Datumaro dataset from COCO person keypoints (val2017) ...")
    dataset = _build_dataset(images_dir, ann_file)
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
