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

    Follows the same contract as Lightning models for pretrained weights,
    preprocessing parameters, and export metadata.

    Args:
        model_name: Model variant identifier (e.g. ``"yolo26n"``).
            Must match a key in :attr:`_pretrained_weights` and
            :attr:`_default_preprocessing_params`.
        label_info: Label metadata used by the engine and DataModule bridge.
        pretrained: Whether to load pretrained weights. When ``True``, the
            model architecture is built from the ``.yaml`` config and
            pretrained weights are loaded from the URL in
            :attr:`_pretrained_weights`.
        imgsz: Image size for training / inference. If ``None``, uses the
            default from :attr:`_default_preprocessing_params`.
        extra_overrides: Extra Ultralytics config forwarded to train/val/export.
    """

    task: ClassVar[str] = ""
    default_model_name: ClassVar[str] = ""
    metric_keys: ClassVar[dict[str, str]] = {}
    trainer_cls: ClassVar[type | None] = None
    validator_cls: ClassVar[type | None] = None

    _pretrained_weights: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        model_name: str | None = None,
        label_info: LabelInfo | None = None,
        *,
        pretrained: bool = True,
        imgsz: int | None = None,
        extra_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name or self.default_model_name
        self.label_info = label_info
        self.pretrained = pretrained
        self.extra_overrides = extra_overrides or {}

        if not self.model_name:
            msg = "model_name must be provided either directly or via the subclass default_model_name attribute."
            raise ValueError(msg)
        if not self.pretrained and self.model_name.endswith(".pt"):
            msg = (
                f"pretrained=False requires a model config, not checkpoint weights: {self.model_name}. "
                "Use a .yaml model definition or a variant name for scratch training."
            )
            raise ValueError(msg)

        # Resolve image size: explicit arg > default from preprocessing params.
        if imgsz is not None:
            self.imgsz = imgsz
        else:
            default_params = self._default_preprocessing_params
            if isinstance(default_params, dict):
                params = default_params.get(self.model_name)
                self.imgsz = params.input_size[0] if params else 640
            else:
                self.imgsz = default_params.input_size[0]

        self._yolo: YOLO | None = None

    @property
    def yolo(self) -> YOLO:
        """Lazily build and return the underlying ``ultralytics.YOLO`` instance."""
        if self._yolo is None:
            self._yolo = self._build_yolo()
        return self._yolo

    def _build_yolo(self) -> YOLO:
        """Create the ``ultralytics.YOLO`` model and optionally load pretrained weights."""
        # Always build from .yaml config for the model architecture.
        yaml_name = self._model_yaml_name
        logger.info(f"Building Ultralytics model: {yaml_name} (task={self.task})")
        yolo = YOLO(yaml_name, task=self.task or None)

        if self.pretrained and self.model_name in self._pretrained_weights:
            weights_url = self._pretrained_weights[self.model_name]
            logger.info(f"Loading pretrained weights: {weights_url}")
            yolo.load(weights_url)

        return yolo

    @property
    def _model_yaml_name(self) -> str:
        """Resolve the YAML config name for model architecture construction.

        Subclasses may override to map ``model_name`` to a different YAML file.
        Default: ``<model_name>.yaml``.
        """
        name = self.model_name
        if name.endswith((".yaml", ".pt")):
            return name
        return f"{name}.yaml"

    def load_checkpoint(self, weights_path: str | Path) -> None:
        """Load weights from a local checkpoint file.

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
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        """Per-model preprocessing parameters.

        Subclasses must override to provide model-specific defaults.
        Returns either a single ``DataInputParams`` or a dict keyed by
        ``model_name`` (like Lightning models).

        YOLO models expect ``[0, 1]`` float input at inference, produced by
        dividing raw uint8 pixels by 255.  ModelAPI's ``InputTransform``
        computes ``(image - mean) / scale``, so:
        - ``mean = (0, 0, 0)`` — no mean subtraction
        - ``std = (255, 255, 255)`` — divide by 255
        """
        return DataInputParams(
            input_size=(self.imgsz, self.imgsz),
            mean=(0.0, 0.0, 0.0),
            std=(255.0, 255.0, 255.0),
        )

    @property
    def data_input_params(self) -> DataInputParams:
        """Resolved data input parameters for this model instance.

        Uses mean/std from :attr:`_default_preprocessing_params` and
        ``self.imgsz`` for the input size (which may be user-overridden).
        """
        default = self._default_preprocessing_params
        if isinstance(default, dict):
            params = default.get(self.model_name)
            if params is None:
                params = next(iter(default.values()))
        else:
            params = default

        # Always use self.imgsz — it may have been overridden by the user.
        if params.input_size != (self.imgsz, self.imgsz):
            return DataInputParams(
                input_size=(self.imgsz, self.imgsz),
                mean=params.mean,
                std=params.std,
            )
        return params

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Task-level export parameters for metadata embedding.

        Subclasses override to set model_type, task_type, and thresholds.
        """
        label_info = self.label_info or LabelInfo(label_names=[], label_ids=[], label_groups=[])
        return TaskLevelExportParameters(
            model_type="YOLO11",
            model_name=self.model_name or "",
            task_type="detection",
            label_info=label_info,
            optimization_config={},
            confidence_threshold=0.25,
            iou_threshold=0.7,
        )

    def __repr__(self) -> str:
        num_classes = self.label_info.num_classes if self.label_info is not None else None
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"num_classes={num_classes}, "
            f"pretrained={self.pretrained})"
        )
