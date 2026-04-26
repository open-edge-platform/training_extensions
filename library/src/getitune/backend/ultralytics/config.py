# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the Ultralytics backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UltralyticsModelConfig:
    """Model section of an Ultralytics recipe.

    Attributes:
        class_path: Fully-qualified path for the model wrapper class.
        model_name: Ultralytics model identifier (e.g. ``"yolo26n.pt"``).
        pretrained: Whether to load pretrained weights.
        imgsz: Input image size for training and inference.
    """

    class_path: str = ""
    model_name: str = ""
    pretrained: bool = True
    imgsz: int = 640


@dataclass
class UltralyticsTrainConfig:
    """Training hyperparameters forwarded to ``yolo.train()``.

    Field names intentionally match Ultralytics train kwargs to minimize
    translation code.
    """

    epochs: int = 100
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    patience: int = 100
    close_mosaic: int = 0

    def to_train_args(self) -> dict[str, Any]:
        """Return kwargs dict for ``yolo.train()``."""
        return {
            "epochs": self.epochs,
            "batch": self.batch,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "box": self.box,
            "cls": self.cls,
            "dfl": self.dfl,
            "patience": self.patience,
            "close_mosaic": self.close_mosaic,
        }


@dataclass
class UltralyticsExportConfig:
    """Export defaults for the Ultralytics backend."""

    format: str = "OPENVINO"
    precision: str = "FP32"


@dataclass
class UltralyticsEngineConfig:
    """Engine section of an Ultralytics recipe."""

    device: str = "auto"


@dataclass
class UltralyticsConfig:
    """Root configuration parsed from an Ultralytics recipe.

    Attributes:
        backend: Must be ``"ultralytics"``.
        task: Task type string matching ``TaskType`` enum values.
        model: Model wrapper configuration.
        engine: Engine configuration.
        training: Training hyperparameters.
        export: Export defaults.
    """

    backend: str = "ultralytics"
    task: str = ""
    model: UltralyticsModelConfig = field(default_factory=UltralyticsModelConfig)
    engine: UltralyticsEngineConfig = field(default_factory=UltralyticsEngineConfig)
    training: UltralyticsTrainConfig = field(default_factory=UltralyticsTrainConfig)
    export: UltralyticsExportConfig = field(default_factory=UltralyticsExportConfig)
