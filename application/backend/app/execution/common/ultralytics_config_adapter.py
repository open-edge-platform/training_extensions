# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application-side adapter for Ultralytics recipe configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from getitune.backend.ultralytics.configurator import UltralyticsConfigurator


class UltralyticsConfigAdapter:
    """Translate Geti application config into an Ultralytics backend config.

    This class intentionally lives in ``application/backend`` because it knows
    about Geti UI ``hyper_parameters`` names and the training service's config
    dict shape. The library ``UltralyticsConfigurator`` remains application-agnostic.
    """

    def __init__(self, configurator: UltralyticsConfigurator) -> None:
        self._configurator = configurator

    @classmethod
    def from_recipe(cls, recipe_path: Path) -> UltralyticsConfigAdapter:
        """Create an adapter from an Ultralytics recipe path."""
        return cls(UltralyticsConfigurator.from_recipe(recipe_path))

    def apply_hyper_parameters(self, hyper_parameters: dict[str, Any] | None) -> None:
        """Apply supported Geti hyperparameter overrides."""
        if not hyper_parameters:
            return

        training = hyper_parameters.get("training", {})
        self._apply_training_parameters(training)
        self._apply_augmentation_parameters(hyper_parameters.get("dataset_preparation", {}).get("augmentation", {}))

    def to_training_config(self) -> dict[str, Any]:
        """Return the backend-tagged config consumed by ``GetiTuneTrainer``."""
        config = self._configurator.config
        return {
            "backend": "ultralytics",
            "task": config.task,
            "model": {
                "class_path": config.model.class_path,
                "init_args": {
                    "model_name": config.model.model_name,
                    "pretrained": config.model.pretrained,
                    "imgsz": config.model.imgsz,
                },
            },
            "engine": {"device": config.engine.device},
            "training": config.training.to_train_args(),
            "export": {
                "format": config.export.format,
                "precision": config.export.precision,
            },
            "data": copy.deepcopy(self._configurator.data_config),
        }

    def _apply_training_parameters(self, training: dict[str, Any]) -> None:
        cfg = self._configurator.config

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
            self._set_subset_batch_size(int(batch_size))

        height = training.get("input_size_height")
        width = training.get("input_size_width")
        if height is not None and width is not None:
            h, w = int(height), int(width)
            cfg.model.imgsz = max(h, w)
            self._configurator.data_config["input_size"] = [h, w]

        early_stopping = training.get("early_stopping")
        if isinstance(early_stopping, dict):
            if early_stopping.get("enable") is False:
                cfg.training.patience = 0
            elif early_stopping.get("enable") is True and early_stopping.get("patience") is not None:
                cfg.training.patience = int(early_stopping["patience"])

    def _apply_augmentation_parameters(self, augmentations: dict[str, Any]) -> None:
        if not augmentations:
            return

        tiling = augmentations.get("tiling")
        if isinstance(tiling, dict):
            self._apply_tiling(tiling)

        flip = augmentations.get("random_horizontal_flip")
        if isinstance(flip, dict):
            self._set_cpu_augmentation(
                class_path="torchvision.transforms.v2.RandomHorizontalFlip",
                enabled=flip.get("enable", True),
                init_args={"p": flip.get("probability", 0.5)},
                insert_before="getitune.data.augmentation.transforms.Resize",
            )

        iou_crop = augmentations.get("iou_random_crop")
        if isinstance(iou_crop, dict):
            self._set_cpu_augmentation(
                class_path="getitune.data.augmentation.transforms.RandomIoUCrop",
                enabled=iou_crop.get("enable", True),
                init_args=None,
                insert_before="getitune.data.augmentation.transforms.Resize",
            )

    def _apply_tiling(self, tiling: dict[str, Any]) -> None:
        data = self._configurator.data_config
        tile_config = data.setdefault(
            "tile_config",
            {
                "enable_tiler": False,
                "enable_adaptive_tiling": False,
            },
        )
        tile_config["enable_tiler"] = bool(tiling.get("enable", False))
        if tile_config["enable_tiler"]:
            tile_size = int(tiling.get("tile_size", 400))
            tile_config["enable_adaptive_tiling"] = bool(tiling.get("enable_adaptive_tiling", True))
            tile_config["tile_size"] = [tile_size, tile_size]
            tile_config["overlap"] = float(tiling.get("tile_overlap", 0.2))

    def _set_subset_batch_size(self, batch_size: int) -> None:
        for subset in ("train_subset", "val_subset", "test_subset"):
            if subset in self._configurator.data_config:
                self._configurator.data_config[subset]["batch_size"] = batch_size

    def _set_cpu_augmentation(
        self,
        class_path: str,
        enabled: bool,
        init_args: dict[str, Any] | None,
        insert_before: str,
    ) -> None:
        train_subset = self._configurator.data_config.get("train_subset", {})
        aug_list = train_subset.setdefault("augmentations_cpu", [])
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
