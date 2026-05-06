# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics instance-segmentation model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo

from .base import UltralyticsModel


class UltralyticsInstSegModel(UltralyticsModel):
    """YOLO instance-segmentation model.

    Supported variants: ``yolo26n-seg``, ``yolo26s-seg``, ``yolo26m-seg``.
    """

    task: ClassVar[str] = "segment"
    default_model_name: ClassVar[str] = "yolo26n-seg"
    trainer_cls: ClassVar[type] = SegmentationTrainer
    validator_cls: ClassVar[type] = SegmentationValidator

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "yolo26n-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n-seg.pt",
        "yolo26s-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s-seg.pt",
        "yolo26m-seg": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m-seg.pt",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults."""
        default = DataInputParams(input_size=(640, 640), mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0))
        return {
            "yolo26n-seg": default,
            "yolo26s-seg": default,
            "yolo26m-seg": default,
        }

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Instance segmentation export parameters.

        Since Ultralytics models are exported without built-in NMS
        (``end2end=False``), we set ``nms_execute=True`` so that ModelAPI
        performs NMS during post-processing.
        """
        label_info = self.label_info or LabelInfo(label_names=[], label_ids=[], label_groups=[])
        return TaskLevelExportParameters(
            model_type="YOLO11-seg",
            model_name=self.model_name or "",
            task_type="instance_segmentation",
            label_info=label_info,
            optimization_config={},
            confidence_threshold=0.25,
            iou_threshold=0.7,
            nms_execute=True,
            agnostic_nms=True,
        )

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
