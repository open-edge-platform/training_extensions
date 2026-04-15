# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Converter for v1 config."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from warnings import warn

from getitune.backend.native.cli.utils import get_otx_root_path
from getitune.tools.auto_configurator import AutoConfigurator
from loguru import logger

RECIPE_PATH = get_otx_root_path() / "recipe"


class ModelStatus(str, Enum):
    """Enum for model status."""

    SPEED = "speed"
    BALANCE = "balance"
    ACCURACY = "accuracy"
    DEPRECATED = "deprecated"
    ACTIVE = "active"


TEMPLATE_ID_MAPPING = {
    # MULTI_CLASS_CLS
    "image-classification-deit-tiny": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "deit_tiny.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    "image-classification-dinov2": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "dino_v2.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    "image-classification-efficientnet-b0": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_b0.yaml",
        "status": ModelStatus.ACTIVE,
        "default": True,
    },
    "image-classification-efficientnet-v2-s": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_v2.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "image-classification-mobilenet-v3-large": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "mobilenet_v3_large.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "image-classification-efficientnet-b3": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_b3.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    # DETECTION
    "object-detection-atss-mobilenet-v2": {
        "recipe_path": RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
        "status": ModelStatus.ACTIVE,
        "default": True,
    },
    "object-detection-ssd-mobilenet-v2": {
        "recipe_path": RECIPE_PATH / "detection" / "ssd_mobilenetv2.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-yolox-x": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_x.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-yolox-l": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_l.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-yolox-s": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_s.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "object-detection-yolox-tiny": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_tiny.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-rt-detr-r50": {
        "recipe_path": RECIPE_PATH / "detection" / "rtdetr_50.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-dfine-m": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_m.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    "object-detection-dfine-l": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_l.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    "object-detection-dfine-x": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_x.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-rfdetr-s": {
        "recipe_path": RECIPE_PATH / "detection" / "rfdetr_small.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-rfdetr-m": {
        "recipe_path": RECIPE_PATH / "detection" / "rfdetr_medium.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-rfdetr-l": {
        "recipe_path": RECIPE_PATH / "detection" / "rfdetr_large.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-dinov3-detr-s": {
        "recipe_path": RECIPE_PATH / "detection" / "deimv2_s.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-dinov3-detr-m": {
        "recipe_path": RECIPE_PATH / "detection" / "deimv2_m.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "object-detection-dinov3-detr-l": {
        "recipe_path": RECIPE_PATH / "detection" / "deimv2_l.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    # INSTANCE_SEGMENTATION
    "instance-segmentation-mask-rcnn-swin-t": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_swint.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "instance-segmentation-mask-rcnn-efficientnet-b2": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_efficientnetb2b.yaml",
        "status": ModelStatus.ACTIVE,
        "default": True,
    },
    "instance-segmentation-rtmdet-tiny": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rtmdet_inst_tiny.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "instance-segmentation-mask-rcnn-resnet50": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "instance-segmentation-rfdetr-s": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_small.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "instance-segmentation-rfdetr-m": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_medium.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    "instance-segmentation-rfdetr-l": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_large.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "instance-segmentation-rfdetr-xl": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_xlarge.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
}


class TransformsUpdater:
    """Handles augmentation updates for the new CPU/GPU augmentation pipeline.

    Maps Geti augmentation names to OTX/kornia/torchvision class paths and the
    pipeline stage (cpu or gpu). Parameters come directly from the Geti model template;
    only a few Geti param names need renaming to match kornia's API.

    Example Geti model template augmentation section::

        random_affine:
            enable: false
            max_rotate_degree: 10.0
            max_translate_ratio: 0.1
            scaling_ratio_range: [0.5, 1.5]
            max_shear_degree: 2.0
        color_jitter:
            enable: false
            brightness: [0.875, 1.125]
            probability: 0.5
    """

    # Geti name -> (class_path, stage)
    # class_paths is a list to match multiple possible implementations in configs
    AUGMENTATION_REGISTRY: ClassVar[dict[str, dict]] = {
        "random_resize_crop": {
            "class_paths": [
                "torchvision.transforms.v2.RandomResizedCrop",
            ],
            "stage": "cpu",
        },
        "random_affine": {
            "class_paths": ["kornia.augmentation.RandomAffine"],
            "stage": "gpu",
        },
        "random_horizontal_flip": {
            "class_paths": ["kornia.augmentation.RandomHorizontalFlip"],
            "stage": "gpu",
        },
        "random_vertical_flip": {
            "class_paths": ["kornia.augmentation.RandomVerticalFlip"],
            "stage": "gpu",
        },
        "gaussian_blur": {
            "class_paths": ["kornia.augmentation.RandomGaussianBlur"],
            "stage": "gpu",
        },
        "gaussian_noise": {
            "class_paths": ["kornia.augmentation.RandomGaussianNoise"],
            "stage": "gpu",
            # kornia.augmentation.RandomGaussianNoise uses ``std``; manifests use ``sigma``
            "param_rename": {"sigma": "std"},
        },
        "color_jitter": {
            "class_paths": ["kornia.augmentation.ColorJiggle"],
            "stage": "gpu",
        },
        "iou_random_crop": {
            "class_paths": ["getitune.data.augmentation.transforms.RandomIoUCrop"],
            "stage": "cpu",
        },
        "random_zoom_out": {
            "class_paths": ["torchvision.transforms.v2.RandomZoomOut"],
            "stage": "cpu",
        },
        "mixup": {
            "class_paths": ["getitune.data.augmentation.transforms.CachedMixUp"],
            "stage": "cpu",
        },
        "mosaic": {
            "class_paths": ["getitune.data.augmentation.transforms.CachedMosaic"],
            "stage": "cpu",
        },
        "random_erasing": {
            "class_paths": ["kornia.augmentation.RandomErasing"],
            "stage": "gpu",
        },
        "random_grayscale": {
            "class_paths": ["kornia.augmentation.RandomGrayscale"],
            "stage": "gpu",
        },
        "random_sharpness": {
            "class_paths": ["kornia.augmentation.RandomSharpness"],
            "stage": "gpu",
        },
    }

    # Geti param name -> kornia/torchvision param name
    PARAM_RENAME: ClassVar[dict[str, str]] = {
        "probability": "p",
        "max_rotate_degree": "degrees",
        "scaling_ratio_range": "scale",
        "crop_ratio_range": "scale",
        "aspect_ratio_range": "ratio",
        "max_translate_ratio": "translate",
        "max_shear_degree": "shear",
    }

    @classmethod
    def update(cls, augmentation_params: dict, config: dict) -> None:  # noqa: C901, PLR0912
        """Update augmentations in the config based on Geti model template.

        For each augmentation in augmentation_params:
        - If enable=True and aug exists in config -> update its parameters
        - If enable=True and aug does NOT exist -> add it with template params
        - If enable=False and aug exists -> remove it
        - If enable=False and aug does NOT exist -> no-op

        Special case: disabling random_resize_crop replaces it with plain Resize.

        Args:
            augmentation_params: Dict mapping Geti aug names to their parameter dicts.
            config: The full OTX config dictionary.
        """
        if not augmentation_params:
            return

        tiling = config["data"].get("tile_config", {}).get("enable_tiler", False)
        train_subset = config["data"]["train_subset"]

        for aug_name, aug_value in augmentation_params.items():
            if aug_name not in cls.AUGMENTATION_REGISTRY:
                if tiling:
                    logger.info("Augmentation '%s' is not applicable in Tiling pipeline", aug_name)
                    continue
                msg = f"Unknown augmentation: '{aug_name}'. Available: {list(cls.AUGMENTATION_REGISTRY.keys())}"
                raise ValueError(msg)

            registry_entry = cls.AUGMENTATION_REGISTRY[aug_name]
            # Work on a copy so we don't mutate the original
            params = dict(aug_value)
            enable = params.pop("enable", True)
            stage_key = f"augmentations_{registry_entry['stage']}"

            # Ensure the stage list exists
            if stage_key not in train_subset:
                train_subset[stage_key] = []

            aug_list = train_subset[stage_key]
            existing_idx = cls._find_augmentation(aug_list, registry_entry["class_paths"])

            if enable:
                init_args = cls._remap_params(params, registry_entry.get("param_rename"))

                if existing_idx is not None:
                    # Update existing augmentation parameters
                    aug_config = aug_list[existing_idx]
                    if "init_args" not in aug_config:
                        aug_config["init_args"] = {}
                    aug_config["init_args"].update(init_args)
                    aug_config.pop("enable", None)
                else:
                    # Add new augmentation with template params
                    new_aug: dict[str, Any] = {"class_path": registry_entry["class_paths"][0]}
                    if init_args:
                        new_aug["init_args"] = init_args
                    insert_idx = cls._get_insert_position(aug_list, registry_entry["stage"])
                    aug_list.insert(insert_idx, new_aug)
            elif existing_idx is not None:
                if aug_name == "random_resize_crop":
                    # Replace crop with simple Resize to keep the pipeline valid
                    aug_list[existing_idx] = {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": "$(input_size)"},
                    }
                else:
                    aug_list.pop(existing_idx)

    @classmethod
    def _remap_params(cls, params: dict, per_aug_rename: dict[str, str] | None = None) -> dict:
        """Rename Geti parameter names to kornia/torchvision names and adjust values.

        1. Rename keys via PARAM_RENAME (probability->p, max_translate_ratio->translate, etc.)
           plus any per-augmentation overrides supplied via per_aug_rename.
        2. Adjust values where kornia expects a different format than a single scalar.

        Args:
            params: Raw Geti parameter dict for the augmentation.
            per_aug_rename: Optional extra rename map applied after PARAM_RENAME.  Use this
                when a kornia class uses a different argument name than the global default
                (e.g. RandomGaussianNoise uses ``std`` while the manifest stores ``sigma``).
        """
        # Step 1: rename keys (global renames + per-augmentation overrides)
        rename_map = {**cls.PARAM_RENAME, **(per_aug_rename or {})}
        init_args: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            init_args[rename_map.get(key, key)] = value

        # Step 2: adjust values to match kornia expected formats
        if "translate" in init_args and not isinstance(init_args["translate"], list):
            v = init_args["translate"]
            init_args["translate"] = [v, v]
        if "shear" in init_args and not isinstance(init_args["shear"], list):
            v = init_args["shear"]
            init_args["shear"] = [-v, v]
        if "kernel_size" in init_args and isinstance(init_args["kernel_size"], int):
            v = init_args["kernel_size"]
            init_args["kernel_size"] = [v, v]

        return init_args

    @staticmethod
    def _find_augmentation(aug_list: list[dict], class_paths: list[str]) -> int | None:
        """Find the index of an augmentation in the list by its class path."""
        for idx, aug_config in enumerate(aug_list):
            if aug_config.get("class_path") in class_paths:
                return idx
        return None

    @staticmethod
    def _get_insert_position(aug_list: list[dict], stage: str) -> int:
        """Determine where to insert a new augmentation.

        GPU: insert before Normalize (should always be last).
        CPU: insert before Resize or at the end.
        """
        if stage == "gpu":
            for idx, aug in enumerate(aug_list):
                if "Normalize" in aug.get("class_path", ""):
                    return idx
        elif stage == "cpu":
            for idx, aug in enumerate(aug_list):
                class_path = aug.get("class_path", "")
                if "Resize" in class_path and "RandomResizedCrop" not in class_path:
                    return idx
        return len(aug_list)

    @staticmethod
    def update_tiling(tiling_dict: dict | None, config: dict) -> None:
        """Update tiling parameters in the config.

        Args:
            tiling_dict: Dict with keys: enable, enable_adaptive_tiling, tile_size, tile_overlap.
            config: The full OTX config dictionary.
        """
        if tiling_dict is None:
            logger.info("Tiling parameters are not provided, skipping update.")
            return

        config["data"]["tile_config"]["enable_tiler"] = tiling_dict["enable"]
        if tiling_dict["enable"]:
            config["data"]["tile_config"]["enable_adaptive_tiling"] = tiling_dict["enable_adaptive_tiling"]
            config["data"]["tile_config"]["tile_size"] = (
                tiling_dict["tile_size"],
                tiling_dict["tile_size"],
            )
            config["data"]["tile_config"]["overlap"] = tiling_dict["tile_overlap"]


class HyperparametersUpdater:
    """Handles training hyperparameter updates (learning rate, batch size, etc.)."""

    @staticmethod
    def update(hyperparameters: dict, config: dict) -> None:  # noqa: C901
        """Update hyperparameters in the config.

        Supported keys:
        - learning_rate: float
        - batch_size: int
        - max_epochs: int (alias for num_iters)
        - early_stopping: dict with keys {enable, patience}
        - input_size: tuple (height, width)

        Args:
            hyperparameters: Dict of hyperparameter updates.
            config: The full OTX config dictionary.
        """
        for key, value in hyperparameters.items():
            if key == "learning_rate":
                HyperparametersUpdater._update_learning_rate(value, config)
            elif key == "batch_size":
                HyperparametersUpdater._update_batch_size(value, config)
            elif key == "max_epochs":
                HyperparametersUpdater._update_max_epochs(value, config)
            elif key == "early_stopping":
                HyperparametersUpdater._update_early_stopping(value, config)
            elif key == "input_size":
                HyperparametersUpdater._update_input_size(value, config)
            elif key == "weight_decay":
                HyperparametersUpdater._update_weight_decay(value, config)
            elif key == "scheduler":
                HyperparametersUpdater._update_scheduler(value, config)
            elif key == "gradient_accumulation":
                HyperparametersUpdater._update_gradient_accumulation(value, config)
            elif key == "gradient_clip":
                HyperparametersUpdater._update_gradient_clip(value, config)
            else:
                logger.warning("Unknown hyperparameter '%s' - skipping update", key)

    @staticmethod
    def _update_learning_rate(param_value: float | None, config: dict) -> None:
        """Update learning rate in the optimizer config."""
        if param_value is None:
            logger.info("Learning rate is not provided, skipping update.")
            return
        optimizer = config["model"]["init_args"]["optimizer"]
        if isinstance(optimizer, dict) and "init_args" in optimizer:
            optimizer["init_args"]["lr"] = param_value
        else:
            warn("Warning: learning_rate is not updated", stacklevel=1)

    @staticmethod
    def _update_batch_size(param_value: int | None, config: dict) -> None:
        """Update batch size for train and val subsets."""
        if param_value is None:
            logger.info("Batch size is not provided, skipping update.")
            return
        config["data"]["train_subset"]["batch_size"] = param_value
        config["data"]["val_subset"]["batch_size"] = param_value

    @staticmethod
    def _update_max_epochs(param_value: int | None, config: dict) -> None:
        """Update max_epochs in the config."""
        if param_value is None:
            logger.info("Max epochs is not provided, skipping update.")
            return
        config["max_epochs"] = param_value

    @staticmethod
    def _update_early_stopping(early_stopping_cfg: dict | None, config: dict) -> None:
        """Update early stopping parameters in the config."""
        if early_stopping_cfg is None:
            logger.info("Early stopping parameters are not provided, skipping update.")
            return

        enable = early_stopping_cfg["enable"]
        patience = early_stopping_cfg.get("patience")

        idx = GetiConfigConverter.get_callback_idx(
            config["callbacks"],
            "getitune.backend.native.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup",
        )
        if not enable and idx > -1:
            config["callbacks"].pop(idx)
            return

        if patience is not None:
            config["callbacks"][idx]["init_args"]["patience"] = patience

    @staticmethod
    def update_tiling(tiling_dict: dict | None, config: dict) -> None:
        """Update tiling parameters in the config."""
        if tiling_dict is None:
            logger.info("Tiling parameters are not provided, skipping update.")
            return

        config["data"]["tile_config"]["enable_tiler"] = tiling_dict["enable"]
        if tiling_dict["enable"]:
            config["data"]["tile_config"]["enable_adaptive_tiling"] = tiling_dict["enable_adaptive_tiling"]
            config["data"]["tile_config"]["tile_size"] = (tiling_dict["tile_size"], tiling_dict["tile_size"])
            config["data"]["tile_config"]["overlap"] = tiling_dict["tile_overlap"]

    @staticmethod
    def _update_input_size(size_value: tuple[int, int] | None, config: dict) -> None:
        """Update input size in the config.

        Args:
            size_value: Tuple of (height, width) or None.
            config: The full OTX config dictionary.
        """
        if size_value is None or any(v is None for v in size_value):
            logger.info("Input size is not provided, skipping update.")
            return
        config["data"]["input_size"] = size_value

    @staticmethod
    def _update_weight_decay(param_value: float | None, config: dict) -> None:
        """Update weight_decay in the optimizer config."""
        if param_value is None:
            logger.info("Weight decay is not provided, skipping update.")
            return
        optimizer = config["model"]["init_args"]["optimizer"]
        if isinstance(optimizer, dict) and "init_args" in optimizer:
            optimizer["init_args"]["weight_decay"] = param_value
        else:
            warn("Warning: weight_decay is not updated", stacklevel=1)

    @staticmethod
    def _update_scheduler(scheduler_cfg: dict | None, config: dict) -> None:
        """Update scheduler parameters in the config."""
        if scheduler_cfg is None:
            logger.info("Scheduler parameters are not provided, skipping update.")
            return

        scheduler = config["model"]["init_args"].get("scheduler")
        if not isinstance(scheduler, dict) or "init_args" not in scheduler:
            warn("Warning: scheduler config not found in recipe, skipping update.", stacklevel=1)
            return

        scheduler_init_args = scheduler["init_args"]

        # Update warmup
        warmup_cfg = scheduler_cfg.get("warmup")
        if warmup_cfg is not None:
            enable_warmup = warmup_cfg.get("enable", False)
            warmup_epochs = warmup_cfg.get("epochs", 0)
            if enable_warmup and warmup_epochs > 0:
                scheduler_init_args["num_warmup_steps"] = warmup_epochs
                scheduler_init_args["warmup_interval"] = "epoch"
            else:
                scheduler_init_args["num_warmup_steps"] = 0

        # Update the main scheduler parameters
        main_scheduler = scheduler_init_args.get("main_scheduler_callable")
        if not isinstance(main_scheduler, dict) or "init_args" not in main_scheduler:
            return

        scheduler_cfg.get("type")
        main_init_args = main_scheduler["init_args"]

        # Update type-specific params regardless of whether type was changed
        factor = scheduler_cfg.get("factor")
        patience = scheduler_cfg.get("patience")
        scheduler_cfg.get("min_lr")

        if "ReduceLROnPlateau" in main_scheduler.get("class_path", ""):
            if factor is not None:
                main_init_args["factor"] = factor
            if patience is not None:
                main_init_args["patience"] = patience

    @staticmethod
    def _update_gradient_clip(gradient_clip_cfg: dict | None, config: dict) -> None:
        """Update gradient clipping in the config.

        The value is stored at ``config["gradient_clip_val"]`` and passed to ``pl.Trainer``.
        """
        if gradient_clip_cfg is None:
            logger.info("Gradient clip parameters are not provided, skipping update.")
            return

        enable = gradient_clip_cfg.get("enable", False)
        if enable:
            max_grad_norm = gradient_clip_cfg.get("max_grad_norm")
            if max_grad_norm is not None:
                config["engine"]["gradient_clip_val"] = max_grad_norm
        else:
            # Explicitly disable gradient clipping
            config["engine"]["gradient_clip_val"] = None

    @staticmethod
    def _update_gradient_accumulation(gradient_accum_cfg: dict | None, config: dict) -> None:
        """Update gradient accumulation in the config.

        The value is stored at ``config["engine"]["accumulate_grad_batches"]``.
        """
        if gradient_accum_cfg is None:
            logger.info("Gradient accumulation parameters are not provided, skipping update.")
            return

        enable = gradient_accum_cfg.get("enable", False)
        if enable:
            batches = gradient_accum_cfg.get("batches", 1)
            if batches > 1:
                config.setdefault("engine", {})
                config["engine"]["accumulate_grad_batches"] = batches
        # Disable accumulation (set to 1 or remove)
        elif "engine" in config and "accumulate_grad_batches" in config.get("engine", {}):
            config["engine"]["accumulate_grad_batches"] = 1


class GetiConfigConverter:
    """Convert Geti model manifest to OTXv2 recipe dictionary.

    Example:
        The following examples show how to use the Converter class.
        We expect a config file with ModelTemplate information in json form.

        Convert template.json to dictionary::

            converter = GetiConfigConverter()
            config = converter.convert("train_config.yaml")

        Instantiate an object from the configuration dictionary::

            engine, train_kwargs = converter.instantiate(
                config=config,
                work_dir="getitune-workspace",
                data_root="tests/assets/detection_coco",
            )

        Train the model::

            engine.train(**train_kwargs)
    """

    @staticmethod
    def convert(config: dict) -> dict:
        """Convert a geti configuration file to a default configuration dictionary.

        Args:
            config (dict): The path to the Geti yaml configuration file.

        Returns:
            dict: The default configuration dictionary.

        """
        hyper_parameters = config["hyper_parameters"]

        model_config_path: Path = TEMPLATE_ID_MAPPING[config["model_manifest_id"]]["recipe_path"]  # type: ignore[assignment]
        # override necessary parameters for config
        tile_enabled = hyper_parameters and hyper_parameters.get("dataset_preparation", {}).get("augmentation", {}).get(
            "tiling",
            {},
        ).get("enable", False)
        if tile_enabled and "_tile" not in model_config_path.stem:
            tile_name = model_config_path.stem + "_tile.yaml"
            model_config_path = model_config_path.parent / tile_name
        # classification task type can't be deducted from template name, try to extract from config
        if (sub_task_type := config["sub_task_type"]) and "_cls" in model_config_path.parent.name:
            model_config_path = RECIPE_PATH / "classification" / sub_task_type.lower() / model_config_path.name
        if model_config_path.suffix != ".yaml":
            model_config_path = model_config_path / ".yaml"
        default_config = AutoConfigurator(model=model_config_path).config
        if hyper_parameters:
            GetiConfigConverter._update_params(default_config, hyper_parameters)
        GetiConfigConverter._remove_unused_key(default_config)
        return default_config

    @staticmethod
    def _get_params(hyperparameters: dict) -> dict:
        """Get configuraable parameters from ModelTemplate config hyperparameters field."""
        param_dict = {}
        for param_name, param_info in hyperparameters.items():
            if isinstance(param_info, dict):
                if "value" in param_info:
                    param_dict[param_name] = param_info["value"]
                else:
                    param_dict = param_dict | GetiConfigConverter._get_params(param_info)

        return param_dict

    @staticmethod
    def _update_params(config: dict, param_dict: dict) -> None:
        """Update params of OTX recipe from Geti configurable params.

        Uses TransformsUpdater and HyperparametersUpdater classes to apply updates
        from the Geti model template to the OTX recipe config.
        """
        augmentation_params = param_dict.get("dataset_preparation", {}).get("augmentation", {})
        tiling = augmentation_params.pop("tiling", None)
        deim_framework = augmentation_params.pop("deim_framework", None)
        training_parameters = param_dict.get("training", {})

        # Update tiling (always applied regardless of DEIM state)
        TransformsUpdater.update_tiling(tiling, config)

        # When DEIM is enabled, the AugmentationSchedulerCallback owns the pipeline;
        # user augmentation overrides must be ignored.
        deim_enabled = deim_framework is True
        if not deim_enabled:
            TransformsUpdater.update(augmentation_params, config)
            if deim_framework is False:
                # User explicitly disabled DEIM -> remove the scheduler callback
                GetiConfigConverter._disable_deim_framework(config)

        # Update training hyperparameters
        hyperparams: dict[str, Any] = {
            "learning_rate": training_parameters.get("learning_rate"),
            "batch_size": training_parameters.get("batch_size"),
            "max_epochs": training_parameters.get("max_epochs"),
            "early_stopping": training_parameters.get("early_stopping"),
            "input_size": (
                training_parameters.get("input_size_height"),
                training_parameters.get("input_size_width"),
            ),
            "weight_decay": training_parameters.get("weight_decay"),
            "scheduler": training_parameters.get("scheduler"),
            "gradient_clip": training_parameters.get("gradient_clip"),
            "gradient_accumulation": training_parameters.get("gradient_accumulation"),
        }
        HyperparametersUpdater.update(hyperparams, config)

    @staticmethod
    def get_callback_idx(callbacks: list, name: str) -> int:
        """Return required callbacks index from callback list."""
        for idx, callback in enumerate(callbacks):
            if callback["class_path"] == name:
                return idx
        return -1

    @staticmethod
    def _disable_deim_framework(config: dict) -> None:
        """Disable the DEIM adaptive augmentation scheduling framework.

        Removes the AugmentationSchedulerCallback from the callbacks list,
        falling back to the static augmentation pipeline defined in
        ``data.train_subset``.
        """
        callbacks = config.get("callbacks", [])
        idx = GetiConfigConverter.get_callback_idx(
            callbacks,
            "getitune.backend.native.callbacks.aug_scheduler.AugmentationSchedulerCallback",
        )
        if idx > -1:
            callbacks.pop(idx)
            logger.info("DEIM framework disabled: removed AugmentationSchedulerCallback")

    @staticmethod
    def _remove_unused_key(config: dict) -> None:
        """Remove unused keys from the config dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        config.pop("config")  # Remove config key that for CLI
        config["data"].pop("__path__", None)  # Remove __path__ key that for CLI overriding
