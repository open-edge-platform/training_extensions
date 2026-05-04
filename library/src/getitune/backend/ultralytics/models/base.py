# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for Ultralytics models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from ultralytics import YOLO

from getitune.backend.lightning.models.base import DataInputParams
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo

logger = logging.getLogger(__name__)


class UltralyticsModel:
    """Wrapper around ``ultralytics.YOLO`` for :class:`UltralyticsEngine`.

    Args:
        model_name: Model identifier (e.g. ``"yolo11s.pt"``).
        label_info: Label metadata used by the engine and DataModule bridge.
        pretrained: Whether checkpoint-style model names are allowed. To train
            from scratch, pass a model config (for example ``"yolo11s.yaml"``)
            with ``pretrained=False``.
        imgsz: Image size for training / inference.
        extra_overrides: Extra Ultralytics config forwarded to train/val/export.
    """

    # Subclass-level defaults.
    task: ClassVar[str] = ""
    default_model_name: ClassVar[str] = ""
    metric_keys: ClassVar[dict[str, str]] = {}
    trainer_cls: ClassVar[type | None] = None
    validator_cls: ClassVar[type | None] = None

    def __init__(
        self,
        model_name: str | None = None,
        label_info: LabelInfo | None = None,
        *,
        pretrained: bool = True,
        imgsz: int = 640,
        extra_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name or self.default_model_name
        self.label_info = label_info
        self.pretrained = pretrained
        self.imgsz = imgsz
        self.extra_overrides = extra_overrides or {}

        if not self.model_name:
            msg = "model_name must be provided either directly or via the subclass default_model_name attribute."
            raise ValueError(msg)
        if not self.pretrained and self.model_name.endswith(".pt"):
            msg = (
                f"pretrained=False requires a model config, not checkpoint weights: {self.model_name}. "
                "Use a .yaml model definition for scratch training."
            )
            raise ValueError(msg)

        self._yolo: YOLO | None = None

    @property
    def yolo(self) -> YOLO:
        """Lazily build and return the underlying ``ultralytics.YOLO`` instance."""
        if self._yolo is None:
            self._yolo = self._build_yolo()
        return self._yolo

    def _build_yolo(self) -> YOLO:
        """Create the ``ultralytics.YOLO`` model."""
        logger.info(f"Building Ultralytics model: {self.model_name} (task={self.task})")
        return YOLO(self.model_name, task=self.task or None)

    def load_checkpoint(self, weights_path: str | Path) -> None:
        """Load pretrained weights from a local checkpoint file.

        The model architecture must already be built (from a ``.yaml`` config).
        Uses :meth:`ultralytics.YOLO.load` to inject checkpoint weights into
        the existing model.

        Args:
            weights_path: Path to a ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(weights_path)
        if not path.exists():
            msg = f"Checkpoint file not found: {path}"
            raise FileNotFoundError(msg)
        logger.info(f"Loading checkpoint weights from: {path}")
        self.yolo.load(str(path))

    @property
    def num_classes(self) -> int | None:
        """Number of classes from ``label_info``, or ``None``."""
        return self.label_info.num_classes if self.label_info is not None else None

    @property
    def export_model_type(self) -> str:
        """ModelAPI ``model_type`` string for exported models.

        Subclasses may override for task-specific model types.
        """
        return "YOLO11"

    @property
    def export_task_type(self) -> str:
        """ModelAPI ``task_type`` string for exported models.

        Subclasses may override for task-specific values.
        """
        return "detection"

    @property
    def data_input_params(self) -> DataInputParams:
        """Data input parameters for model preprocessing.

        YOLO models expect [0, 1] float input (the /255 happens externally),
        so ``mean=(0, 0, 0)`` and ``std=(1, 1, 1)`` — identity normalisation.
        """
        return DataInputParams(
            input_size=(self.imgsz, self.imgsz),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Task-level export parameters for metadata embedding.

        Returns a :class:`TaskLevelExportParameters` with detection defaults.
        Subclasses may override for task-specific values.
        """
        label_info = self.label_info or LabelInfo(label_names=[], label_ids=[], label_groups=[])
        return TaskLevelExportParameters(
            model_type=self.export_model_type,
            model_name=self.model_name or "",
            task_type=self.export_task_type,
            label_info=label_info,
            optimization_config={},
            confidence_threshold=0.25,
            iou_threshold=0.7,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"num_classes={self.num_classes}, "
            f"pretrained={self.pretrained})"
        )
