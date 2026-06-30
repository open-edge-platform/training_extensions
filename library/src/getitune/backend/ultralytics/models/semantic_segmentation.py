# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics semantic segmentation model wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.semantic_segmentation import SemanticSegmentationTrainer
from getitune.backend.ultralytics.validators.semantic_segmentation import SemanticSegmentationValidator
from getitune.config.data import IntensityConfig
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo, LabelInfoTypes, SegLabelInfo

from .base import UltralyticsModel

if TYPE_CHECKING:
    from getitune.backend.ultralytics.exporter import UltralyticsModelExporter


class UltralyticsSemanticSegModel(UltralyticsModel):
    """YOLO semantic segmentation model.

    Supported variants: ``yolo26n-sem``, ``yolo26s-sem``, ``yolo26m-sem``,
    ``yolo26l-sem``, ``yolo26x-sem``.
    """

    task: ClassVar[str] = "semantic"
    trainer_cls: ClassVar[type] = SemanticSegmentationTrainer
    validator_cls: ClassVar[type] = SemanticSegmentationValidator

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "yolo26n-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt",
        "yolo26s-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem.pt",
        "yolo26m-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-sem.pt",
        "yolo26l-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-sem.pt",
        "yolo26x-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-sem.pt",
    }

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> SegLabelInfo:
        """Normalize label_info to a ``SegLabelInfo`` instance.

        Semantic segmentation needs the ignore_index field for proper handling
        of void pixels during metric computation and loss computation.
        """
        if isinstance(label_info, SegLabelInfo):
            return label_info
        if isinstance(label_info, dict):
            return SegLabelInfo(**label_info)
        if isinstance(label_info, int):
            return SegLabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, (list, tuple)) and all(isinstance(name, str) for name in label_info):
            names = list(label_info)
            return SegLabelInfo(
                label_names=names,
                label_groups=[names],
                label_ids=[str(i) for i in range(len(names))],
            )
        if isinstance(label_info, LabelInfo):
            return SegLabelInfo(
                label_names=label_info.label_names,
                label_groups=label_info.label_groups,
                label_ids=label_info.label_ids,
            )
        raise TypeError(label_info)

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        All YOLO26-sem models use 512x512 input with identity mean/std (no
        additional normalization after intensity scaling to [0, 1]).
        """
        default = DataInputParams(
            input_size=(512, 512),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
        )
        return {
            "yolo26n-sem": default,
            "yolo26s-sem": default,
            "yolo26m-sem": default,
            "yolo26l-sem": default,
            "yolo26x-sem": default,
        }

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Semantic segmentation export parameters."""
        label_info = self.label_info or SegLabelInfo(label_names=[], label_ids=[], label_groups=[])
        return TaskLevelExportParameters(
            model_type="YOLO-sem",
            model_name=self.model_name,
            task_type="semantic_segmentation",
            label_info=label_info,
            optimization_config={},
            confidence_threshold=0.0,
            return_soft_prediction=True,
            blur_strength=0,
            nms_execute=False,
        )

    @property
    def _exporter(self) -> UltralyticsModelExporter:
        """Build and return the model exporter with standard (non-letterbox) resize."""
        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        return UltralyticsModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
        )

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mIoU": "val/mIoU",
        "metrics/pixel_acc": "val/pixel_accuracy",
        "train/ce_loss": "train/ce_loss",
        "train/dice_loss": "train/dice_loss",
        "train/aux_loss": "train/aux_loss",
        "lr/pg0": "lr",
    }
