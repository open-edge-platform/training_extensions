# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics detection model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.validators.detection import DetectionValidator

from .base import UltralyticsModel


class UltralyticsDetectionModel(UltralyticsModel):
    """YOLO detection model (default: ``yolo11s.yaml``)."""

    task: ClassVar[str] = "detect"
    default_model_name: ClassVar[str] = "yolo11s.yaml"
    trainer_cls: ClassVar[type] = DetectionTrainer
    validator_cls: ClassVar[type] = DetectionValidator

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/dfl_loss": "train/dfl_loss",
    }
