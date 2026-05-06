# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-local configurator for Ultralytics recipes.

This module is the single entry point for creating, configuring, and
serialising Ultralytics training configs.  The application converter calls
:meth:`UltralyticsConfigurator.convert` (or the lower-level
:meth:`apply_hyper_parameters` + :meth:`to_config_dict`) — no
backend-specific knowledge is required on the caller side.
"""

from __future__ import annotations

import copy
import importlib
import logging
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from getitune.types.device import DeviceType
from getitune.types.task import TaskType

from .config import (
    UltralyticsConfig,
    UltralyticsEngineConfig,
    UltralyticsExportConfig,
    UltralyticsModelConfig,
    UltralyticsTrainConfig,
)
from .engine import UltralyticsEngine
from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from getitune.data.module import DataModule
    from getitune.types import PathLike
    from getitune.types.label import LabelInfo

logger = logging.getLogger(__name__)

# Task string → default model wrapper class path.
_TASK_TO_DEFAULT_CLASS_PATH: dict[str, str] = {
    TaskType.DETECTION.value: "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel",
    TaskType.INSTANCE_SEGMENTATION.value: (
        "getitune.backend.ultralytics.models.instance_segmentation.UltralyticsInstSegModel"
    ),
}

_SUPPORTED_TASKS: frozenset[str] = frozenset(_TASK_TO_DEFAULT_CLASS_PATH)


class UltralyticsConfigurator:
    """Parse Ultralytics recipes, apply hyper-parameters, and construct model / engine instances.

    This class consolidates recipe loading, hyper-parameter translation
    (Geti → Ultralytics naming), serialisation, and model/engine
    instantiation in a single place.
    """

    def __init__(
        self,
        config: UltralyticsConfig,
        data_config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._data_config: dict[str, Any] = data_config or {}
        self._validate_backend()
        self._validate_task()


    @classmethod
    def from_recipe(cls, recipe_path: PathLike) -> UltralyticsConfigurator:
        """Load an Ultralytics recipe YAML.

        Args:
            recipe_path: Path to the recipe YAML file.

        Returns:
            Configured ``UltralyticsConfigurator`` instance.

        Raises:
            FileNotFoundError: If the recipe file does not exist.
            ValueError: If the recipe has an invalid backend or task.
        """
        path = Path(recipe_path)
        if not path.exists():
            msg = f"Recipe file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:  # noqa: PTH123
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            msg = f"Recipe must be a YAML mapping, got {type(raw).__name__}"
            raise TypeError(msg)

        config = cls._parse_raw_config(raw)

        # Resolve the data section (string reference or inline dict).
        data_config = cls._resolve_data_config(raw.get("data"), path.parent)

        # Apply inline data overrides (mirrors Lightning `overrides.data`).
        overrides_data = raw.get("overrides", {}).get("data", {})
        if overrides_data:
            _deep_merge(data_config, overrides_data)

        return cls(config, data_config=data_config)

    @classmethod
    def from_config_dict(cls, raw: dict[str, Any]) -> UltralyticsConfigurator:
        """Build a configurator from an already-loaded recipe dictionary."""
        config = cls._parse_raw_config(raw)
        data_raw = raw.get("data")
        data_config = dict(data_raw) if isinstance(data_raw, dict) else {}
        return cls(config, data_config=data_config)


    @property
    def config(self) -> UltralyticsConfig:
        """The parsed Ultralytics configuration."""
        return self._config

    @property
    def data_config(self) -> dict[str, Any]:
        """The resolved DataModule-compatible data configuration."""
        return self._data_config


    @staticmethod
    def is_ultralytics_recipe(recipe_path: Path) -> bool:
        """Return whether *recipe_path* declares the Ultralytics backend."""
        with recipe_path.open() as f:
            recipe = yaml.safe_load(f)
        return isinstance(recipe, dict) and recipe.get("backend") == "ultralytics"

    @classmethod
    def convert(cls, recipe_path: Path, hyper_parameters: dict | None = None) -> dict:
        """Load an Ultralytics recipe, apply *hyper_parameters*, return a training config dict.

        This is the single entry-point the application converter should call.

        Args:
            recipe_path: Path to the Ultralytics recipe YAML.
            hyper_parameters: Standard Geti hyper_parameter dict (same keys as
                Lightning: ``learning_rate``, ``batch_size``, ``max_epochs``, etc.).

        Returns:
            A serialised training config dict consumable by
            :meth:`from_config_dict`.
        """
        configurator = cls.from_recipe(recipe_path)
        if hyper_parameters:
            configurator.apply_hyper_parameters(hyper_parameters)
        return configurator.to_config_dict()

    def apply_hyper_parameters(self, hyper_parameters: dict) -> None:
        """Apply standard Geti *hyper_parameters* to this configurator.

        Only handles the ``training`` section (learning rate, epochs, batch size,
        etc.).  Augmentation and tiling updates are **not** applied here — they
        flow through the standard ``TransformsUpdater`` in the application layer,
        identically to Lightning recipes.
        """
        training = hyper_parameters.get("training", {})
        if training:
            self._apply_training_params(training)

    def to_config_dict(self) -> dict:
        """Serialise this configurator to the dict consumed by the trainer."""
        cfg = self._config
        data = copy.deepcopy(self._data_config)

        # Ensure tile_config exists with defaults so the shared
        # TransformsUpdater.update_tiling() can write to it.
        data.setdefault("tile_config", {"enable_tiler": False, "enable_adaptive_tiling": False})

        return {
            "backend": "ultralytics",
            "task": cfg.task,
            "model": {
                "class_path": cfg.model.class_path,
                "init_args": {
                    "model_name": cfg.model.model_name,
                    "pretrained": cfg.model.pretrained,
                    "imgsz": cfg.model.imgsz,
                },
            },
            "engine": {"device": cfg.engine.device},
            "training": cfg.training.to_train_args(),
            "export": {
                "format": cfg.export.format,
                "precision": cfg.export.precision,
            },
            "data": data,
        }


    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Merge supported overrides into the current config.

        Accepts dot-separated keys (``"training.epochs"``) or nested dicts
        (``{"training": {"epochs": 50}}``).  Unknown keys raise ``ValueError``.
        """
        if not overrides:
            return

        flat = _flatten_overrides(overrides)
        for key, value in flat.items():
            self._apply_single_override(key, value)


    def create_model(
        self,
        label_info: LabelInfo,
        weights_path: PathLike | None = None,
    ) -> UltralyticsModel:
        """Instantiate the model from recipe config.

        The model is always built from the recipe's ``.yaml`` architecture
        config (no pretrained weights downloaded by Ultralytics).  When
        *weights_path* is provided (e.g. from ``BaseWeightsService``), the
        checkpoint is loaded into the model after construction.

        Args:
            label_info: Label metadata for the dataset.
            weights_path: Optional path to a local ``.pt`` checkpoint
                (downloaded externally) that will be loaded into the model.

        Returns:
            Configured ``UltralyticsModel`` subclass instance.
        """
        model_cls = self._resolve_model_wrapper_cls()

        model = model_cls(
            model_name=self._config.model.model_name,
            label_info=label_info,
            pretrained=self._config.model.pretrained,
            imgsz=self._config.model.imgsz,
        )

        if weights_path is not None:
            model.load_checkpoint(weights_path)

        return model

    def create_engine(
        self,
        model: UltralyticsModel,
        data: DataModule | PathLike,
        work_dir: PathLike,
        device: str | DeviceType = DeviceType.auto,
        **engine_kwargs: object,
    ) -> UltralyticsEngine:
        """Instantiate the engine from recipe config.

        Training defaults from the recipe are passed as train-only defaults.

        Args:
            model: Ultralytics model wrapper.
            data: DataModule or filesystem data-root path.
            work_dir: Directory for training artefacts.
            device: Device specification (overrides recipe default when not
                ``"auto"``).
            **engine_kwargs: Extra kwargs forwarded to the engine constructor.
        """
        # Use caller-specified device when it is not "auto"; fall back to recipe.
        resolved_device: str | DeviceType = device
        if str(device) == str(DeviceType.auto) or str(device) == "auto":
            resolved_device = self._config.engine.device

        return UltralyticsEngine(
            model=model,
            data=data,
            work_dir=work_dir,
            device=resolved_device,
            train_args=self._config.training.to_train_args(),
            export_args={
                "confidence_threshold": self._config.export.confidence_threshold,
                "iou_threshold": self._config.export.iou_threshold,
            },
            **engine_kwargs,
        )


    def _apply_training_params(self, training: dict) -> None:
        """Map standard Geti training params to Ultralytics config fields."""
        cfg = self._config

        lr = training.get("learning_rate")
        if lr is not None:
            cfg.training.lr0 = float(lr)

        weight_decay = training.get("weight_decay")
        if weight_decay is not None:
            cfg.training.weight_decay = float(weight_decay)

        max_epochs = training.get("max_epochs")
        if max_epochs is not None:
            cfg.training.epochs = int(max_epochs)

        batch_size = training.get("batch_size")
        if batch_size is not None:
            cfg.training.batch = int(batch_size)
            # Propagate to data subsets so DataModule sees the same value.
            for subset in ("train_subset", "val_subset", "test_subset"):
                if subset in self._data_config:
                    self._data_config[subset]["batch_size"] = int(batch_size)

        height = training.get("input_size_height")
        width = training.get("input_size_width")
        if height is not None and width is not None:
            h, w = int(height), int(width)
            cfg.model.imgsz = max(h, w)
            self._data_config["input_size"] = [h, w]

        self._apply_early_stopping(cfg, training.get("early_stopping"))

    @staticmethod
    def _apply_early_stopping(
        cfg: UltralyticsConfig,
        early_stopping: dict | None,
    ) -> None:
        """Map early-stopping params to Ultralytics ``patience``."""
        if not isinstance(early_stopping, dict):
            return
        if early_stopping.get("enable") is False:
            cfg.training.patience = 0
        elif early_stopping.get("enable") is True and early_stopping.get("patience") is not None:
            cfg.training.patience = int(early_stopping["patience"])


    def _validate_backend(self) -> None:
        if self._config.backend != "ultralytics":
            msg = f"Expected backend 'ultralytics', got '{self._config.backend}'"
            raise ValueError(msg)

    def _validate_task(self) -> None:
        if self._config.task not in _SUPPORTED_TASKS:
            msg = f"Unsupported task '{self._config.task}'. Supported: {sorted(_SUPPORTED_TASKS)}"
            raise ValueError(msg)


    def _resolve_model_wrapper_cls(self) -> type[UltralyticsModel]:
        """Return the model wrapper class from recipe ``class_path`` or task."""
        class_path = self._config.model.class_path
        if not class_path:
            class_path = _TASK_TO_DEFAULT_CLASS_PATH.get(self._config.task, "")

        if not class_path:
            msg = f"Cannot resolve model class for task '{self._config.task}'"
            raise ValueError(msg)

        cls = _import_class(class_path)
        if not issubclass(cls, UltralyticsModel):
            msg = f"model.class_path must resolve to an UltralyticsModel subclass, got '{class_path}'"
            raise TypeError(msg)
        return cls


    def _apply_single_override(self, key: str, value: object) -> None:
        """Apply a single dot-separated override to the config."""
        parts = key.split(".", 1)
        expected_parts = 2
        if len(parts) != expected_parts:
            msg = f"Override key must be section.field, got '{key}'"
            raise ValueError(msg)

        section_name, field_name = parts

        section_map: dict[str, object] = {
            "training": self._config.training,
            "model": self._config.model,
            "engine": self._config.engine,
            "export": self._config.export,
        }

        section = section_map.get(section_name)
        if section is None:
            msg = f"Unknown override section: '{section_name}'"
            raise ValueError(msg)

        if not hasattr(section, field_name):
            msg = f"Unknown override: '{key}'"
            raise ValueError(msg)

        setattr(section, field_name, value)


    @classmethod
    def _parse_raw_config(cls, raw: dict[str, Any]) -> UltralyticsConfig:
        """Parse raw YAML dict into typed config dataclasses."""
        model_raw = raw.get("model", {})
        engine_raw = raw.get("engine", {})
        training_raw = raw.get("training", {})
        export_raw = raw.get("export", {})

        # Model: flatten class_path + init_args.
        model_init = model_raw.get("init_args", {})
        model_config = UltralyticsModelConfig(
            class_path=model_raw.get("class_path", ""),
            model_name=model_init.get("model_name", ""),
            pretrained=model_init.get("pretrained", True),
            imgsz=model_init.get("imgsz", 640),
        )

        # Engine: accept both flat and nested init_args.
        engine_init = engine_raw.get("init_args", engine_raw)
        engine_config = UltralyticsEngineConfig(
            device=engine_init.get("device", "auto"),
        )

        # Training: only accept known fields.
        training_fields = {f.name for f in fields(UltralyticsTrainConfig)}
        training_config = UltralyticsTrainConfig(
            **{k: v for k, v in training_raw.items() if k in training_fields},
        )

        # Export: only accept known fields.
        export_fields = {f.name for f in fields(UltralyticsExportConfig)}
        export_config = UltralyticsExportConfig(
            **{k: v for k, v in export_raw.items() if k in export_fields},
        )

        return UltralyticsConfig(
            backend=raw.get("backend", "ultralytics"),
            task=raw.get("task", ""),
            model=model_config,
            engine=engine_config,
            training=training_config,
            export=export_config,
        )

    @staticmethod
    def _resolve_data_config(
        data_ref: str | dict[str, Any] | None,
        recipe_dir: Path,
    ) -> dict[str, Any]:
        """Resolve a data reference (relative YAML path) or inline dict."""
        if data_ref is None:
            return {}
        if isinstance(data_ref, dict):
            return dict(data_ref)

        # String reference to a base YAML.
        data_path = (recipe_dir / str(data_ref)).resolve()
        if not data_path.exists():
            msg = f"Referenced data config not found: {data_path}"
            raise FileNotFoundError(msg)

        with open(data_path) as f:  # noqa: PTH123
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}




def _import_class(class_path: str) -> type:
    """Import a class from a fully-qualified dotted path."""
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        msg = f"Invalid class_path: '{class_path}' (no module prefix)"
        raise ValueError(msg)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        msg = f"Cannot import module '{module_path}' from class_path '{class_path}'"
        raise ValueError(msg) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        msg = f"Class '{class_name}' not found in module '{module_path}'"
        raise ValueError(msg)

    return cls


def _flatten_overrides(overrides: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_overrides(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Deep-merge *overrides* into *base* in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
