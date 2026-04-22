# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics instance segmentation model for getitune."""

from __future__ import annotations

from typing import ClassVar

from .base import UltralyticsModel


class UltralyticsInstSegModel(UltralyticsModel):
    """Ultralytics YOLO model for instance segmentation.

    Wraps a YOLO segmentation model (e.g. ``yolo11s-seg.pt``) for use
    with the getitune engine.  Default model is ``yolo11s-seg.pt``.

    The model integrates with getitune's data pipeline: images arrive as
    ``float32 CHW [0,1]`` tensors from the CPU augmentation pipeline and
    are passed directly to Ultralytics without uint8 conversion.
    """

    task: str = "segment"
    default_model_name: str = "yolo11s-seg.pt"

    # Maps Ultralytics metric keys -> getitune-style metric names.
    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/mAP50(M)": "val/mask_map_50",
        "metrics/mAP50-95(M)": "val/mask_map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/dfl_loss": "train/dfl_loss",
        "train/seg_loss": "train/seg_loss",
    }
