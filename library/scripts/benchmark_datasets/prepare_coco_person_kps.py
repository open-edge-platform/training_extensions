#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare a COCO person keypoints benchmark dataset.

Downloads COCO 2017 val split person keypoint annotations and the corresponding
images. Each person annotation is converted into a single, person-cropped
``KeypointSample`` (top-down representation) so the model receives images
already centered on a single person — matching how RTMPose and other top-down
pose estimators are trained.

The val2017 split is partitioned deterministically into train / val / test
(70 / 15 / 15) so the benchmark is fully self-contained.

Source: https://cocodataset.org/
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import KeypointCategories, LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from PIL import Image as PILImage
from torchvision import tv_tensors

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args
from getitune.data.entity.sample import KeypointSample

if TYPE_CHECKING:
    from pathlib import Path

# COCO 2017 val annotations (includes person_keypoints)
_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# COCO 2017 val images
_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

# COCO person keypoint skeleton definition (17 keypoints)
_KEYPOINT_NAMES: tuple[str, ...] = (
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
_NUM_KEYPOINTS = len(_KEYPOINT_NAMES)

# Only use the "person" category
_PERSON_CATEGORY = "person"

# Fraction of val images to use for train / validation / test (deterministic split)
_TRAIN_FRACTION = 0.7
_VAL_FRACTION = 0.15

# Top-down crop parameters
# - We expand the COCO person bbox by this factor to include surrounding
#   context (limbs, hair) so keypoints near the bbox edge are not clipped.
_BBOX_PADDING = 1.25
# - Final crop is resized to this fixed size to bound dataset size on disk.
#   The training-time recipe further resizes to its model input size.
_CROP_SIZE = 256
# - Reject persons whose original bbox is too small (likely background/noise).
_MIN_BBOX_SIDE = 24
# - Require this many visible keypoints in the original annotation.
_MIN_VISIBLE_KEYPOINTS = 5


def _expand_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Expand a COCO ``(x, y, w, h)`` bbox by *padding* and clamp to image bounds.

    Returns ``(left, top, right, bottom)`` integer pixel coordinates suitable
    for ``PIL.Image.crop``.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * padding
    new_h = h * padding
    left = round(max(0.0, cx - new_w / 2.0))
    top = round(max(0.0, cy - new_h / 2.0))
    right = round(min(float(img_w), cx + new_w / 2.0))
    bottom = round(min(float(img_h), cy + new_h / 2.0))
    return left, top, right, bottom


def _build_dataset(images_dir: Path, ann_file: Path) -> Dataset:
    """Parse COCO person keypoints annotations and build a Datumaro dataset.

    Each valid person annotation becomes one ``KeypointSample`` whose image is
    a fixed-size crop centered on the person and whose keypoints are expressed
    in the cropped/resized image's coordinate frame.
    """
    with ann_file.open() as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    person_cat_ids = {cid for cid, name in cat_id_to_name.items() if name == _PERSON_CATEGORY}
    if not person_cat_ids:
        msg = "No 'person' category found in annotations"
        raise ValueError(msg)

    img_lookup = {img["id"]: img for img in coco["images"]}

    # Collect valid person annotations with enough visible keypoints
    valid_anns: list[dict] = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in person_cat_ids:
            continue
        if ann.get("iscrowd"):
            continue
        kps = ann.get("keypoints")
        if not kps or len(kps) != _NUM_KEYPOINTS * 3:
            continue
        if sum(1 for i in range(2, len(kps), 3) if kps[i] > 0) < _MIN_VISIBLE_KEYPOINTS:
            continue
        bbox = ann.get("bbox")
        if not bbox or bbox[2] < _MIN_BBOX_SIDE or bbox[3] < _MIN_BBOX_SIDE:
            continue
        valid_anns.append(ann)

    # Sort by annotation id for deterministic ordering
    valid_anns.sort(key=lambda a: a["id"])

    num_anns = len(valid_anns)
    train_count = int(num_anns * _TRAIN_FRACTION)
    val_count = int(num_anns * (_TRAIN_FRACTION + _VAL_FRACTION))

    dataset: Dataset = Dataset(
        KeypointSample,
        categories={
            "label": LabelCategories(labels=(_PERSON_CATEGORY,)),
            "keypoints": KeypointCategories(labels=_KEYPOINT_NAMES),
        },
    )

    skipped_missing_image = 0
    skipped_empty_crop = 0

    for ann_idx, ann in enumerate(valid_anns):
        img_info = img_lookup[ann["image_id"]]
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            skipped_missing_image += 1
            continue

        img_w = int(img_info["width"])
        img_h = int(img_info["height"])
        bx, by, bw, bh = (float(v) for v in ann["bbox"])
        left, top, right, bottom = _expand_bbox(bx, by, bw, bh, img_w, img_h, _BBOX_PADDING)
        if right <= left or bottom <= top:
            skipped_empty_crop += 1
            continue
        crop_w = right - left
        crop_h = bottom - top

        # Crop the person out of the image and resize to a fixed square so
        # disk size and model input are both predictable.
        with PILImage.open(img_path) as pil_img:
            rgb = pil_img.convert("RGB")
            crop = rgb.crop((left, top, right, bottom)).resize(
                (_CROP_SIZE, _CROP_SIZE),
                PILImage.BILINEAR,  # type: ignore[attr-defined]
            )
        img_np = np.asarray(crop, dtype=np.uint8)
        img_chw = torch.from_numpy(img_np.transpose(2, 0, 1).copy())
        image = tv_tensors.Image(img_chw)

        # Transform keypoints into the crop's coordinate frame, then into the
        # resized crop. Visibility flag is preserved (clamped to 0/1 by the
        # dataset's _get_item_impl).
        kps_arr = np.asarray(ann["keypoints"], dtype=np.float32).reshape(_NUM_KEYPOINTS, 3)
        scale_x = _CROP_SIZE / float(crop_w)
        scale_y = _CROP_SIZE / float(crop_h)
        kps_arr[:, 0] = (kps_arr[:, 0] - left) * scale_x
        kps_arr[:, 1] = (kps_arr[:, 1] - top) * scale_y
        # Mark keypoints that fell outside the (padded) crop as not visible so
        # the model isn't penalized for occluded/clipped joints.
        out_of_crop = (
            (kps_arr[:, 0] < 0) | (kps_arr[:, 0] >= _CROP_SIZE) | (kps_arr[:, 1] < 0) | (kps_arr[:, 1] >= _CROP_SIZE)
        )
        kps_arr[out_of_crop, 2] = 0
        keypoints = torch.from_numpy(kps_arr)

        if ann_idx < train_count:
            subset = Subset.TRAINING
        elif ann_idx < val_count:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TESTING

        dataset.append(
            KeypointSample(
                image=image,
                label=torch.tensor([0], dtype=torch.long),
                keypoints=keypoints,
                dm_image_info=DmImageInfo(width=_CROP_SIZE, height=_CROP_SIZE),
                subset=subset,
            ),
        )

    if skipped_missing_image:
        print(f"  Skipped {skipped_missing_image} annotations (image file missing).")
    if skipped_empty_crop:
        print(f"  Skipped {skipped_empty_crop} annotations (empty/degenerate crop).")

    return dataset


def main() -> None:
    """Download COCO person keypoints, convert to Datumaro format, and save."""
    args = parse_args(description="Prepare the coco_person_kps benchmark dataset.")

    ann_archive = download(_ANNOTATIONS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_annotations.zip")
    img_archive = download(_VAL_IMAGES_URL, dest_dir=args.archive_dir, filename=f"{args.name}_val2017.zip")

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

    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
