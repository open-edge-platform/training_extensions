# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics instance-segmentation model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator

from .base import UltralyticsModel


class UltralyticsInstSegModel(UltralyticsModel):
    """YOLO instance-segmentation model (default: ``yolo11s-seg.pt``)."""

    task: ClassVar[str] = "segment"
    default_model_name: ClassVar[str] = "yolo11s-seg.pt"
    trainer_cls: ClassVar[type] = SegmentationTrainer
    validator_cls: ClassVar[type] = SegmentationValidator

    @property
    def export_task_type(self) -> str:
        """ModelAPI ``task_type`` for instance segmentation."""
        return "instance_segmentation"

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
