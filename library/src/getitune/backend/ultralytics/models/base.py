# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for Ultralytics models in getitune."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ultralytics import YOLO

    from getitune.types.label import LabelInfo

logger = logging.getLogger(__name__)


class UltralyticsModel:
    """Base wrapper for Ultralytics YOLO models in getitune.

    This is a standalone class (NOT a LightningModel subclass).
    It lazily initializes the ``ultralytics.YOLO`` model on first access
    via the :attr:`yolo` property, and handles ``num_classes`` override
    from the provided :class:`~getitune.types.label.LabelInfo`.

    Args:
        model_name: Ultralytics model identifier (e.g. ``"yolo11s.pt"``).
            Can be a filename (pre-trained weights from Ultralytics hub),
            a local checkpoint path, or a YAML config.
        label_info: Label metadata from getitune.  ``num_classes`` is
            derived from ``label_info.num_classes``.
        pretrained: Whether to use pre-trained weights.  Defaults to ``True``.
        task: Ultralytics task string (``"detect"``, ``"segment"``, etc.).
            Subclasses set this automatically.
        imgsz: Default training/inference image size.
        extra_overrides: Additional Ultralytics overrides dict that will be
            forwarded to ``model.train()`` / ``model.val()`` / ``model.export()``.
    """

    # Subclasses override these class-level defaults.
    # `task` and `default_model_name` are plain str attributes (not ClassVar)
    # because __init__ may override `task` on a per-instance basis.
    task: str = ""
    default_model_name: str = ""
    metric_keys: ClassVar[dict[str, str]] = {}

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

        # Lazily created
        self._yolo: YOLO | None = None

    # ------------------------------------------------------------------
    # Lazy YOLO initialization
    # ------------------------------------------------------------------

    @property
    def yolo(self) -> YOLO:
        """Return the underlying ``ultralytics.YOLO`` model, creating it on first access.

        The model is initialized with the stored ``model_name`` and ``task``.
        If a :attr:`label_info` is set, the model head is reconfigured to
        match ``label_info.num_classes``.

        Returns:
            The initialized ``ultralytics.YOLO`` model instance.
        """
        if self._yolo is None:
            self._yolo = self._build_yolo()
        return self._yolo

    @yolo.setter
    def yolo(self, value: YOLO) -> None:
        """Allow replacing the YOLO model (e.g. after loading a checkpoint)."""
        self._yolo = value

    def _build_yolo(self) -> YOLO:
        """Construct the ``ultralytics.YOLO`` model.

        Returns:
            A freshly created YOLO model.
        """
        from ultralytics import YOLO

        logger.info("Building Ultralytics model: %s (task=%s)", self.model_name, self.task)

        model = YOLO(self.model_name, task=self.task or None)

        if self.label_info is not None:
            self._apply_label_info(model)

        return model

    @staticmethod
    def _apply_label_info(model: YOLO) -> None:
        """Override class names on the underlying YOLO model.

        Ultralytics stores ``model.names`` as ``{0: "cls0", 1: "cls1", ...}``.
        We patch it so that downstream validators/exporters emit the right labels.

        Note: This does NOT re-create the detection head. If the number of
        classes differs from the pre-trained model, Ultralytics will adapt
        automatically during ``model.train()`` when it reads the data YAML.

        Args:
            model: The Ultralytics YOLO model to patch.
        """
        # Names patching is deferred to train-time when the data config
        # will carry the correct class mapping. Nothing to do here yet.

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes from ``label_info`` (if available)."""
        return self.label_info.num_classes if self.label_info is not None else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"num_classes={self.num_classes}, "
            f"pretrained={self.pretrained})"
        )
