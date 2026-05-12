# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configurator for Ultralytics recipes."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import yaml
from jsonargparse import ArgumentParser, Namespace

from getitune.types.device import DeviceType
from getitune.types.task import TaskType

from .engine import UltralyticsEngine
from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from getitune.data.module import DataModule
    from getitune.types import PathLike
    from getitune.types.label import LabelInfo

_SUPPORTED_TASKS: frozenset[str] = frozenset((TaskType.DETECTION.value, TaskType.INSTANCE_SEGMENTATION.value))
ConfigValue: TypeAlias = str | int | float | bool | None | dict[str, "ConfigValue"] | list["ConfigValue"]


class Configurator:
    """Load Ultralytics recipes and instantiate backend objects."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = copy.deepcopy(config)
        if self._config.get("backend") != "ultralytics":
            msg = f"Expected backend 'ultralytics', got '{self._config.get('backend')}'"
            raise ValueError(msg)
        if self._config.get("task") not in _SUPPORTED_TASKS:
            msg = f"Unsupported task '{self._config.get('task')}'. Supported: {sorted(_SUPPORTED_TASKS)}"
            raise ValueError(msg)

    @classmethod
    def from_recipe(cls, recipe_path: PathLike) -> Configurator:
        """Load an Ultralytics recipe YAML."""
        path = Path(recipe_path)
        if not path.exists():
            msg = f"Recipe file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:  # noqa: PTH123
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            msg = f"Recipe must be a YAML mapping, got {type(config).__name__}"
            raise TypeError(msg)

        config = copy.deepcopy(config)
        data_config = cls._resolve_data_config(config.get("data"), path.parent)
        overrides_data = config.pop("overrides", {}).get("data", {})
        if overrides_data:
            _deep_merge(data_config, overrides_data)
        config["data"] = data_config
        return cls(config)

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> Configurator:
        """Build a configurator from an already-loaded configuration dictionary."""
        return cls(config)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration dictionary."""
        return self._config

    @property
    def data_config(self) -> dict[str, Any]:
        """Resolved DataModule-compatible data configuration."""
        data = self._config.setdefault("data", {})
        if not isinstance(data, dict):
            msg = f"Expected resolved data config to be a dict, got {type(data).__name__}"
            raise TypeError(msg)
        return data

    @classmethod
    def convert(cls, recipe_path: Path, hyper_parameters: dict | None = None) -> dict:
        """Load a recipe, apply hyper-parameters, and return a config dict."""
        configurator = cls.from_recipe(recipe_path)
        if hyper_parameters:
            configurator.apply_hyper_parameters(hyper_parameters)
        return configurator.to_config_dict()

    def apply_hyper_parameters(self, hyper_parameters: dict) -> None:
        """Apply standard Geti hyper-parameters."""
        training = hyper_parameters.get("training", {})
        if not training:
            return

        train_cfg = self._config.setdefault("training", {})
        model_init = self._config.setdefault("model", {}).setdefault("init_args", {})

        if (lr := training.get("learning_rate")) is not None:
            train_cfg["lr0"] = float(lr)
        if (weight_decay := training.get("weight_decay")) is not None:
            train_cfg["weight_decay"] = float(weight_decay)
        if (max_epochs := training.get("max_epochs")) is not None:
            train_cfg["epochs"] = int(max_epochs)
        if (batch_size := training.get("batch_size")) is not None:
            train_cfg["batch"] = int(batch_size)
            for subset in ("train_subset", "val_subset", "test_subset"):
                if subset in self.data_config:
                    self.data_config[subset]["batch_size"] = int(batch_size)

        height = training.get("input_size_height")
        width = training.get("input_size_width")
        if height is not None and width is not None:
            h, w = int(height), int(width)
            model_init["imgsz"] = max(h, w)
            self.data_config["input_size"] = [h, w]

        early_stopping = training.get("early_stopping")
        if isinstance(early_stopping, dict):
            if early_stopping.get("enable") is False:
                train_cfg["patience"] = 0
            elif early_stopping.get("enable") is True and early_stopping.get("patience") is not None:
                train_cfg["patience"] = int(early_stopping["patience"])

    def to_config_dict(self) -> dict:
        """Return the config consumed by the application trainer.

        Places ``max_epochs`` at the top level so the application trainer
        can access it uniformly regardless of backend (Lightning stores it
        top-level via jsonargparse CLI resolution).
        """
        config = copy.deepcopy(self._config)
        data = config.setdefault("data", {})
        if isinstance(data, dict):
            data.setdefault("tile_config", {"enable_tiler": False, "enable_adaptive_tiling": False})

        training = config.get("training", {})
        if "epochs" in training:
            config.setdefault("max_epochs", training["epochs"])

        return config

    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Merge supported dot-path overrides into the config."""
        if not overrides:
            return
        for key, value in _flatten_overrides(overrides).items():
            _set_dot_path(self._config, key, value)

    def create_model(self, label_info: LabelInfo, weights_path: PathLike | None = None) -> UltralyticsModel:
        """Instantiate the configured Ultralytics model via jsonargparse."""
        model_config = copy.deepcopy(self._config["model"])
        if "class_path" not in model_config:
            msg = "Model config must include class_path"
            raise ValueError(msg)
        model_config.setdefault("init_args", {})["label_info"] = label_info.as_dict()

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(UltralyticsModel, "model", required=False, fail_untyped=False)
        model: UltralyticsModel = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

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
        """Instantiate the configured Ultralytics engine."""
        resolved_device: str | DeviceType = device
        if str(device) == str(DeviceType.auto) or str(device) == "auto":
            resolved_device = _engine_device(self._config.get("engine", {}))

        export = self._config.get("export", {})
        return UltralyticsEngine(
            model=model,
            data=data,
            work_dir=work_dir,
            device=resolved_device,
            train_args=copy.deepcopy(self._config.get("training", {})),
            export_args={
                "confidence_threshold": export.get("confidence_threshold", 0.25),
                "iou_threshold": export.get("iou_threshold", 0.5),
            },
            **engine_kwargs,
        )

    @staticmethod
    def _resolve_data_config(data_ref: str | dict[str, Any] | None, recipe_dir: Path) -> dict[str, Any]:
        """Resolve a data reference or inline data mapping."""
        if data_ref is None:
            return {}
        if isinstance(data_ref, dict):
            return copy.deepcopy(data_ref)

        data_path = (recipe_dir / str(data_ref)).resolve()
        if not data_path.exists():
            msg = f"Referenced data config not found: {data_path}"
            raise FileNotFoundError(msg)

        with open(data_path) as f:  # noqa: PTH123
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}


def _engine_device(engine_config: dict[str, Any]) -> str:
    init_args = engine_config.get("init_args")
    if isinstance(init_args, dict):
        return str(init_args.get("device", "auto"))
    return str(engine_config.get("device", "auto"))


def _set_dot_path(config: dict[str, Any], key: str, value: ConfigValue) -> None:
    parts = key.split(".")
    if len(parts) < 2:
        msg = f"Override key must be a dot path, got '{key}'"
        raise ValueError(msg)

    current = config
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            msg = f"Cannot override nested key '{key}' because '{part}' is not a mapping"
            raise TypeError(msg)
        current = next_value
    current[parts[-1]] = value


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
