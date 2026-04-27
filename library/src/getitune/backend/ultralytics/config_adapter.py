# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapter that bridges standard training hyper_parameters to Ultralytics config.

This module provides the conversion layer between the application's
hyper_parameter format (``learning_rate``, ``batch_size``, ``max_epochs``, etc.)
and the Ultralytics-specific config fields (``lr0``, ``batch``, ``epochs``, etc.).

The application should only call :meth:`UltralyticsConfigAdapter.convert` or
:meth:`UltralyticsConfigAdapter.apply_hyper_parameters` — no backend-specific
knowledge is required on the caller side.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import yaml

from .config import UltralyticsConfig
from .configurator import UltralyticsConfigurator

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class UltralyticsConfigAdapter:
    """Adapt standard Geti hyper_parameters to Ultralytics config format.

    This adapter ensures the application does not need to know about
    Ultralytics-specific parameter names (``lr0``, ``batch``, ``patience``, etc.).
    It accepts the same hyper_parameter dictionary that Lightning models use and
    translates it to Ultralytics equivalents.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_ultralytics_recipe(recipe_path: Path) -> bool:
        """Return whether *recipe_path* declares the Ultralytics backend."""
        with recipe_path.open() as f:
            recipe = yaml.safe_load(f)
        return isinstance(recipe, dict) and recipe.get("backend") == "ultralytics"

    @staticmethod
    def convert(recipe_path: Path, hyper_parameters: dict | None = None) -> dict:
        """Load an Ultralytics recipe, apply *hyper_parameters*, return a training config dict.

        This is the single entry-point the application converter should call.

        Args:
            recipe_path: Path to the Ultralytics recipe YAML.
            hyper_parameters: Standard Geti hyper_parameter dict (same keys as
                Lightning: ``learning_rate``, ``batch_size``, ``max_epochs``, etc.).

        Returns:
            A serialised training config dict consumable by
            :class:`UltralyticsConfigurator.from_config_dict`.
        """
        configurator = UltralyticsConfigurator.from_recipe(recipe_path)
        if hyper_parameters:
            UltralyticsConfigAdapter.apply_hyper_parameters(configurator, hyper_parameters)
        return UltralyticsConfigAdapter.to_config_dict(configurator)

    @staticmethod
    def apply_hyper_parameters(
        configurator: UltralyticsConfigurator,
        hyper_parameters: dict,
    ) -> None:
        """Apply standard Geti *hyper_parameters* to an Ultralytics configurator.

        Handles the ``training`` section (learning rate, epochs, batch size, etc.)
        and the ``dataset_preparation.augmentation`` section (tiling, flips, crops).
        """
        training = hyper_parameters.get("training", {})
        if training:
            UltralyticsConfigAdapter._apply_training_params(configurator, training)

        augmentations = hyper_parameters.get("dataset_preparation", {}).get("augmentation", {})
        if augmentations:
            UltralyticsConfigAdapter._apply_augmentations(configurator, augmentations)

    @staticmethod
    def to_config_dict(configurator: UltralyticsConfigurator) -> dict:
        """Serialise a configurator to the dict consumed by the trainer."""
        cfg = configurator.config
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
            "data": copy.deepcopy(configurator.data_config),
        }

    # ------------------------------------------------------------------
    # Internal: training params
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_training_params(
        configurator: UltralyticsConfigurator,
        training: dict,
    ) -> None:
        """Map standard Geti training params to Ultralytics config fields."""
        cfg = configurator.config

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
                if subset in configurator.data_config:
                    configurator.data_config[subset]["batch_size"] = int(batch_size)

        height = training.get("input_size_height")
        width = training.get("input_size_width")
        if height is not None and width is not None:
            h, w = int(height), int(width)
            cfg.model.imgsz = max(h, w)
            configurator.data_config["input_size"] = [h, w]

        UltralyticsConfigAdapter._apply_early_stopping(cfg, training.get("early_stopping"))

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

    # ------------------------------------------------------------------
    # Internal: augmentations
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_augmentations(
        configurator: UltralyticsConfigurator,
        augmentations: dict,
    ) -> None:
        """Apply augmentation overrides to the Ultralytics data config."""
        tiling = augmentations.get("tiling")
        if isinstance(tiling, dict):
            data = configurator.data_config
            tile_config = data.setdefault(
                "tile_config",
                {"enable_tiler": False, "enable_adaptive_tiling": False},
            )
            tile_config["enable_tiler"] = bool(tiling.get("enable", False))
            if tile_config["enable_tiler"]:
                tile_size = int(tiling.get("tile_size", 400))
                tile_config["enable_adaptive_tiling"] = bool(tiling.get("enable_adaptive_tiling", True))
                tile_config["tile_size"] = [tile_size, tile_size]
                tile_config["overlap"] = float(tiling.get("tile_overlap", 0.2))

        flip = augmentations.get("random_horizontal_flip")
        if isinstance(flip, dict):
            UltralyticsConfigAdapter._set_cpu_augmentation(
                configurator,
                class_path="torchvision.transforms.v2.RandomHorizontalFlip",
                enabled=flip.get("enable", True),
                init_args={"p": flip.get("probability", 0.5)},
                insert_before="getitune.data.augmentation.transforms.Resize",
            )

        iou_crop = augmentations.get("iou_random_crop")
        if isinstance(iou_crop, dict):
            UltralyticsConfigAdapter._set_cpu_augmentation(
                configurator,
                class_path="getitune.data.augmentation.transforms.RandomIoUCrop",
                enabled=iou_crop.get("enable", True),
                init_args=None,
                insert_before="getitune.data.augmentation.transforms.Resize",
            )

    @staticmethod
    def _set_cpu_augmentation(
        configurator: UltralyticsConfigurator,
        class_path: str,
        enabled: bool,
        init_args: dict[str, Any] | None,
        insert_before: str,
    ) -> None:
        """Add, remove, or update a CPU augmentation in the Ultralytics data config."""
        train_subset = configurator.data_config.get("train_subset", {})
        aug_list: list[dict[str, Any]] = train_subset.setdefault("augmentations_cpu", [])
        existing_idx = next((idx for idx, aug in enumerate(aug_list) if aug.get("class_path") == class_path), None)

        if not enabled:
            if existing_idx is not None:
                aug_list.pop(existing_idx)
            return

        aug_config: dict[str, Any] = {"class_path": class_path}
        if init_args is not None:
            aug_config["init_args"] = {k: v for k, v in init_args.items() if v is not None}

        if existing_idx is not None:
            aug_list[existing_idx] = aug_config
            return

        insert_idx = next(
            (idx for idx, aug in enumerate(aug_list) if aug.get("class_path") == insert_before), len(aug_list)
        )
        aug_list.insert(insert_idx, aug_config)
