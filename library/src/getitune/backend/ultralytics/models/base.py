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
    """Wrapper around ``ultralytics.YOLO`` for use with :class:`UltralyticsEngine`.

    The YOLO model is created lazily on the first access to :attr:`yolo`.

    Args:
        model_name: Model identifier, e.g. ``"yolo11s.pt"``, a local path,
            or a YAML config.
        label_info: Label metadata; ``num_classes`` is derived from it.
        pretrained: Load pre-trained weights.
        task: Ultralytics task (``"detect"``, ``"segment"``, ...).
            Subclasses set a sensible default.
        imgsz: Training / inference image size.
        extra_overrides: Extra Ultralytics config passed through to
            ``train()`` / ``val()`` / ``export()``.
    """

    # Subclass defaults.  `task` and `default_model_name` are plain attrs
    # so that __init__ can override them per-instance.
    task: str = ""
    default_model_name: str = ""
    metric_keys: ClassVar[dict[str, str]] = {}
    trainer_cls: ClassVar[type | None] = None

    def __init__(
        self,
        model_name: str | None = None,
        label_info: LabelInfo | None = None,
        *,
        pretrained: bool = True,
        task: str | None = None,
        imgsz: int = 640,
        extra_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name or self.default_model_name
        self.label_info = label_info
        self.pretrained = pretrained
        self.imgsz = imgsz
        self.extra_overrides = extra_overrides or {}

        if task is not None:
            self.task = task

        if not self.model_name:
            msg = "model_name must be provided either directly or via the subclass default_model_name attribute."
            raise ValueError(msg)

        self._yolo: YOLO | None = None

    @property
    def yolo(self) -> YOLO:
        """Lazily build and return the underlying ``ultralytics.YOLO`` instance."""
        if self._yolo is None:
            self._yolo = self._build_yolo()
        return self._yolo

    @yolo.setter
    def yolo(self, value: YOLO) -> None:
        """Replace the YOLO instance (e.g. after loading a checkpoint)."""
        self._yolo = value

    def _build_yolo(self) -> YOLO:
        """Create the ``ultralytics.YOLO`` model."""
        logger.info(f"Building Ultralytics model: {self.model_name} (task={self.task})")
        model = YOLO(self.model_name, task=self.task or None)

        if self.label_info is not None:
            self._apply_label_info(model)

        return model

    @staticmethod
    def _apply_label_info(model: YOLO) -> None:
        """Patch class names on ``model``.

        The actual head adaptation happens inside ``model.train()`` when
        Ultralytics reads the data config, so this is a no-op for now.

        Args:
            model: YOLO model to patch.
        """

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
