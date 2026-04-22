# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics detection model for getitune."""

from __future__ import annotations

from typing import ClassVar

from .base import UltralyticsModel


class UltralyticsDetectionModel(UltralyticsModel):
    """Ultralytics YOLO model for object detection.

    Wraps a YOLO detection model (e.g. ``yolo11s.pt``) for use with the
    getitune engine.  Default model is ``yolo11s.pt``.

    The model integrates with getitune's data pipeline: images arrive as
    ``float32 CHW [0,1]`` tensors from the CPU augmentation pipeline and
    are passed directly to Ultralytics without uint8 conversion.
    """

    task: str = "detect"
    default_model_name: str = "yolo11s.pt"

    # Maps Ultralytics metric keys -> getitune-style metric names.
    # The actual mapping is used by UltralyticsEngine.train() /
    # UltralyticsEngine.test() to translate returned metrics.
    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/dfl_loss": "train/dfl_loss",
    }
