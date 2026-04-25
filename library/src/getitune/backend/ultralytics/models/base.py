# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for Ultralytics models."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ultralytics import YOLO

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

    @property
    def num_classes(self) -> int | None:
        """Number of classes from ``label_info``, or ``None``."""
        return self.label_info.num_classes if self.label_info is not None else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"num_classes={self.num_classes}, "
            f"pretrained={self.pretrained})"
        )
