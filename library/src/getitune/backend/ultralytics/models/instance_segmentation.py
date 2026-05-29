# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics instance-segmentation model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator
from getitune.config.data import IntensityConfig
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo

from .base import UltralyticsModel


class UltralyticsInstSegModel(UltralyticsModel):
    """YOLO instance-segmentation model.

    Supported variants: ``yolo26n-seg``, ``yolo26s-seg``, ``yolo26m-seg``.
    """

    task: ClassVar[str] = "segment"
    trainer_cls: ClassVar[type] = SegmentationTrainer
    validator_cls: ClassVar[type] = SegmentationValidator

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "yolo26n-seg": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
        "yolo26s-seg": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
        "yolo26m-seg": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        Intensity scaling (divide by 255 for uint8) is handled by
        ``IntensityConfig``. Mean/std are identity (no additional
        normalization after intensity scaling).
        """
        default = DataInputParams(
            input_size=(640, 640),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
        )
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
        conf = self._export_args.get("confidence_threshold")
        if conf is None:
            conf = self.extra_overrides.get("conf", 0.25)
        iou = self._export_args.get("iou_threshold")
        if iou is None:
            iou = self.extra_overrides.get("iou", 0.5)
        return TaskLevelExportParameters(
            model_type="YOLO11-seg",
            model_name=self.model_name,
            task_type="instance_segmentation",
            label_info=label_info,
            optimization_config={},
            confidence_threshold=float(conf),
            iou_threshold=float(iou),
            nms_execute=True,
        )

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/mAP50(M)": "val/mask_map_50",
        "metrics/mAP50-95(M)": "val/mask_map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/loss_bbox",
        "train/cls_loss": "train/loss_cls",
        "train/dfl_loss": "train/loss_dfl",
        "train/seg_loss": "train/loss_mask",
    }
