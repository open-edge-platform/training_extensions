# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics detection model."""

from __future__ import annotations

from typing import ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.validators.detection import DetectionValidator
from getitune.config.data import IntensityConfig

from .base import UltralyticsModel


class UltralyticsDetectionModel(UltralyticsModel):
    """YOLO detection model.

    Supported variants: ``yolo26n``, ``yolo26s``, ``yolo26m``.
    """

    task: ClassVar[str] = "detect"
    trainer_cls: ClassVar[type] = DetectionTrainer
    validator_cls: ClassVar[type] = DetectionValidator

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "yolo26n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.pt",
        "yolo26s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s.pt",
        "yolo26m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt",
    }

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        All YOLO26 detection models use 640x640 input. The intensity scaling
        (divide by 255 for uint8) is handled by ``IntensityConfig`` and
        embedded into the exported model via ``intensity_mode``. The mean/std
        here are identity since no additional normalization is needed after
        intensity scaling.
        """
        default = DataInputParams(
            input_size=(640, 640),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
        )
        return {
            "yolo26n": default,
            "yolo26s": default,
            "yolo26m": default,
        }

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mAP50(B)": "val/map_50",
        "metrics/mAP50-95(B)": "val/map",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/dfl_loss": "train/dfl_loss",
    }
