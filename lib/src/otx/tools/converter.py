# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Converter for v1 config."""

from __future__ import annotations

import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import Any
from warnings import warn

import yaml
from jsonargparse import ArgumentParser, Namespace

from otx.backend.native.cli.utils import get_otx_root_path
from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.config.data import SamplerConfig, SubsetConfig, TileConfig
from otx.data.module import OTXDataModule
from otx.engine import Engine, create_engine
from otx.tools.auto_configurator import AutoConfigurator
from otx.types import PathLike

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
    "Custom_Image_Classification_DeiT-Tiny": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "deit_tiny.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Image_Classification_EfficinetNet-B0": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_b0.yaml",
        "status": ModelStatus.BALANCE,
        "default": True,
    },
    "Custom_Image_Classification_EfficientNet-V2-S": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_v2.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    "Custom_Image_Classification_MobileNet-V3-large-1x": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "mobilenet_v3_large.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "Custom_Image_Classification_EfficientNet-B3": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "tv_efficientnet_b3.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Image_Classification_EfficientNet-V2-L": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "tv_efficientnet_v2_l.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Custom_Image_Classification_MobileNet-V3-small": {
        "recipe_path": RECIPE_PATH / "classification" / "multi_class_cls" / "tv_mobilenet_v3_small.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    # DETECTION
    "Custom_Object_Detection_Gen3_ATSS": {
        "recipe_path": RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
        "status": ModelStatus.BALANCE,
        "default": True,
    },
    "Object_Detection_ResNeXt101_ATSS": {
        "recipe_path": RECIPE_PATH / "detection" / "atss_resnext101.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Custom_Object_Detection_Gen3_SSD": {
        "recipe_path": RECIPE_PATH / "detection" / "ssd_mobilenetv2.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Object_Detection_YOLOX_X": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_x.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Object_Detection_YOLOX_L": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_l.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Object_Detection_YOLOX_S": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_s.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "Custom_Object_Detection_YOLOX": {
        "recipe_path": RECIPE_PATH / "detection" / "yolox_tiny.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Object_Detection_RTDetr_18": {
        "recipe_path": RECIPE_PATH / "detection" / "rtdetr_18.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Object_Detection_RTDetr_50": {
        "recipe_path": RECIPE_PATH / "detection" / "rtdetr_50.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Object_Detection_RTDetr_101": {
        "recipe_path": RECIPE_PATH / "detection" / "rtdetr_101.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Object_Detection_RTMDet_tiny": {
        "recipe_path": RECIPE_PATH / "detection" / "rtmdet_tiny.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Object_Detection_DFine_X": {
        "recipe_path": RECIPE_PATH / "detection" / "dfine_x.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    "Object_Detection_Deim_DFine_M": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_m.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Object_Detection_Deim_DFine_L": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_l.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Object_Detection_Deim_DFine_X": {
        "recipe_path": RECIPE_PATH / "detection" / "deim_dfine_x.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    # INSTANCE_SEGMENTATION
    "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_swint.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_efficientnetb2b.yaml",
        "status": ModelStatus.SPEED,
        "default": True,
    },
    "Custom_Instance_Segmentation_RTMDet_tiny": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "rtmdet_inst_tiny.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Instance_Segmentation_MaskRCNN_ResNet50_v2": {
        "recipe_path": RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50_tv.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    # ROTATED_DETECTION
    "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50": {
        "recipe_path": RECIPE_PATH / "rotated_detection" / "maskrcnn_r50.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B": {
        "recipe_path": RECIPE_PATH / "rotated_detection" / "maskrcnn_efficientnetb2b.yaml",
        "status": ModelStatus.SPEED,
        "default": True,
    },
    "Rotated_Detection_MaskRCNN_ResNet50_V2": {
        "recipe_path": RECIPE_PATH / "rotated_detection" / "maskrcnn_r50_v2.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    # SEMANTIC_SEGMENTATION
    "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
        "status": ModelStatus.BALANCE,
        "default": True,
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "litehrnet_s.yaml",
        "status": ModelStatus.SPEED,
        "default": False,
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "litehrnet_x.yaml",
        "status": ModelStatus.DEPRECATED,
        "default": False,
    },
    "Custom_Semantic_Segmentation_SegNext_t": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "segnext_t.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Semantic_Segmentation_SegNext_s": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "segnext_s.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Semantic_Segmentation_SegNext_B": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "segnext_b.yaml",
        "status": ModelStatus.ACTIVE,
        "default": False,
    },
    "Custom_Semantic_Segmentation_DINOV2_S": {
        "recipe_path": RECIPE_PATH / "semantic_segmentation" / "dino_v2.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    # ANOMALY
    "ote_anomaly_padim": {
        "recipe_path": RECIPE_PATH / "anomaly" / "padim.yaml",
        "status": ModelStatus.SPEED,
        "default": True,
    },
    "ote_anomaly_stfpm": {
        "recipe_path": RECIPE_PATH / "anomaly" / "stfpm.yaml",
        "status": ModelStatus.BALANCE,
        "default": False,
    },
    "ote_anomaly_uflow": {
        "recipe_path": RECIPE_PATH / "anomaly" / "uflow.yaml",
        "status": ModelStatus.ACCURACY,
        "default": False,
    },
    # KEYPOINT_DETECTION
    "Keypoint_Detection_RTMPose_Tiny": {
        "recipe_path": RECIPE_PATH / "keypoint_detection" / "rtmpose_tiny.yaml",
        "status": ModelStatus.SPEED,
        "default": True,
    },
}


def update_learning_rate(param_value: float | None, config: dict) -> None:
    """Update learning rate in the config."""
    if param_value is None:
        logging.info("Learning rate is not provided, skipping update.")
        return
    optimizer = config["model"]["init_args"]["optimizer"]
    if isinstance(optimizer, dict) and "init_args" in optimizer:
        optimizer["init_args"]["lr"] = param_value
    else:
        warn("Warning: learning_rate is not updated", stacklevel=1)


def update_num_iters(param_value: int | None, config: dict) -> None:
    """Update max_epochs in the config."""
    if param_value is None:
        logging.info("Max epochs is not provided, skipping update.")
        return
    config["max_epochs"] = param_value


def update_batch_size(param_value: int | None, config: dict) -> None:
    """Update batch size in the config."""
    if param_value is None:
        logging.info("Batch size is not provided, skipping update.")
        return
    config["data"]["train_subset"]["batch_size"] = param_value
    config["data"]["val_subset"]["batch_size"] = param_value


def update_early_stopping(early_stopping_cfg: dict | None, config: dict) -> None:
    """Update early stopping parameters in the config."""
    if early_stopping_cfg is None:
        logging.info("Early stopping parameters are not provided, skipping update.")
        return

    enable = early_stopping_cfg["enable"]
    patience = early_stopping_cfg["patience"]

    idx = GetiConfigConverter.get_callback_idx(
        config["callbacks"],
        "otx.backend.native.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup",
    )
    if not enable and idx > -1:
        config["callbacks"].pop(idx)
        return

    config["callbacks"][idx]["init_args"]["patience"] = patience


def update_tiling(tiling_dict: dict | None, config: dict) -> None:
    """Update tiling parameters in the config."""
    if tiling_dict is None:
        logging.info("Tiling parameters are not provided, skipping update.")
        return

    config["data"]["tile_config"]["enable_tiler"] = tiling_dict["enable"]
    if tiling_dict["enable"]:
        config["data"]["tile_config"]["enable_adaptive_tiling"] = tiling_dict["adaptive_tiling"]
        config["data"]["tile_config"]["tile_size"] = (
            tiling_dict["tile_size"],
            tiling_dict["tile_size"],
        )
        config["data"]["tile_config"]["overlap"] = tiling_dict["tile_overlap"]


def update_input_size(height: int | None, width: int | None, config: dict) -> None:
    """Update input size in the config."""
    if height is None or width is None:
        logging.info("Input size is not provided, skipping update.")
        return
    config["data"]["input_size"] = (height, width)


def update_augmentations(augmentation_params: dict, config: dict) -> None:
    """Update augmentations in the config.

    Example:
        augmentation_params = {
            random_affine = {
                "enable": True,
                "scaling_ratio_range": [0.1, 2.0]
            },
            gaussian_blur = {
                "enable": True,
                "kernel_size": 5
            }
            ...
        }
    """
    if not augmentation_params:
        return

    tiling = config["data"]["tile_config"]["enable_tiler"]
    # this list maps Geti user frendly naming to OTX aug classes
    augs_mapping_list = {
        "random_resize_crop": [
            "otx.data.transform_libs.torchvision.EfficientNetRandomCrop",
            "otx.data.transform_libs.torchvision.RandomResizedCrop",
        ],
        "random_affine": ["otx.data.transform_libs.torchvision.RandomAffine"],
        "topdown_affine": ["otx.data.transform_libs.torchvision.TopdownAffine"],
        "random_horizontal_flip": ["otx.data.transform_libs.torchvision.RandomFlip"],
        "random_vertical_flip": ["torchvision.transforms.v2.RandomVerticalFlip"],
        "gaussian_blur": ["otx.data.transform_libs.torchvision.RandomGaussianBlur"],
        "gaussian_noise": ["otx.data.transform_libs.torchvision.RandomGaussianNoise"],
        "color_jitter": ["torchvision.transforms.v2.RandomPhotometricDistort"],
        "photometric_distort": ["otx.data.transform_libs.torchvision.PhotoMetricDistortion"],
        "iou_random_crop": [
            "otx.data.transform_libs.torchvision.MinIoURandomCrop",
            "otx.data.transform_libs.torchvision.RandomIoUCrop",
        ],
        "random_zoom_out": ["torchvision.transforms.v2.RandomZoomOut"],
        "hsv_random_aug": ["otx.data.transform_libs.torchvision.YOLOXHSVRandomAug"],
        "mixup": ["otx.data.transform_libs.torchvision.CachedMixUp"],
        "mosaic": ["otx.data.transform_libs.torchvision.CachedMosaic"],
    }

    for aug_name, aug_value in augmentation_params.items():
        aug_classes = augs_mapping_list[aug_name]
        found = False
        for aug_config in config["data"]["train_subset"]["transforms"]:
            if aug_config["class_path"] in aug_classes:
                found = True
                if "init_args" not in aug_config:
                    aug_config["init_args"] = {}
                if aug_name == "random_resize_crop" and not aug_value["enable"]:
                    # if random crop is disabled -> change this augmentation to simple Resize
                    aug_config["class_path"] = "otx.data.transform_libs.torchvision.Resize"
                    break
                if "TopdownAffine" in aug_config["class_path"]:
                    affine_transforms_prob = aug_value.pop("probability", 1.0)
                    if affine_transforms_prob is not None:
                        aug_config["init_args"]["probability"] = affine_transforms_prob if aug_value["enable"] else 0.0
                        if aug_config["init_args"]["probability"] < 0.7:
                            for val_aug_cfg in config["data"]["val_subset"]["transforms"]:
                                if "Pad" in val_aug_cfg["class_path"]:
                                    val_aug_cfg["enable"] = False

                    break

                aug_config["enable"] = aug_value.pop("enable")
                for parameter in aug_value:
                    value = aug_value[parameter]
                    if value is not None:
                        override_parameter = (
                            "p"
                            if parameter == "probability" and "torchvision.transforms.v2" in aug_config["class_path"]
                            else parameter
                        )  # Geti consistency fix
                        aug_config["init_args"][override_parameter] = value
                break

        if not found and not tiling:
            msg = f"Augmentation {aug_name} is not found for this model."
            raise ValueError(msg)
        logging.info("This augmentation is not applicable in Tiling pipeline")


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
                work_dir="otx-workspace",
                data_root="tests/assets/car_tree_bug",
            )

        Train the model::

            engine.train(**train_kwargs)
    """

    @staticmethod
    def convert(config: dict) -> dict:
        """Convert a geti configuration file to a default configuration dictionary.

        Args:
            config (dict): The path to the Geti yaml configuration file.
            task (OTXTaskType | None): Value to override the task.

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
        default_config = AutoConfigurator(model_config_path=model_config_path).config
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
        """Update params of OTX recipe from Geit configurable params."""
        augmentation_params = param_dict.get("dataset_preparation", {}).get("augmentation", {})
        tiling = augmentation_params.pop("tiling", None)
        training_parameters = param_dict.get("training", {})

        update_tiling(tiling, config)
        update_augmentations(augmentation_params, config)
        update_learning_rate(training_parameters.get("learning_rate", None), config)
        update_batch_size(training_parameters.get("batch_size", None), config)
        update_num_iters(training_parameters.get("max_epochs", None), config)
        update_early_stopping(training_parameters.get("early_stopping", None), config)
        update_input_size(
            training_parameters.get("input_size_height", None),
            training_parameters.get("input_size_width", None),
            config,
        )

    @staticmethod
    def get_callback_idx(callbacks: list, name: str) -> int:
        """Return required callbacks index from callback list."""
        for idx, callback in enumerate(callbacks):
            if callback["class_path"] == name:
                return idx
        return -1

    @staticmethod
    def _remove_unused_key(config: dict) -> None:
        """Remove unused keys from the config dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        config.pop("config")  # Remove config key that for CLI
        config["data"].pop("__path__", None)  # Remove __path__ key that for CLI overriding

    @staticmethod
    def instantiate_datamodule(config: dict, data_root: PathLike | None = None, **kwargs) -> OTXDataModule:
        """Instantiate an OTXDataModule with arrow data format."""
        config.update(kwargs)

        # Instantiate datamodule
        data_config = config.pop("data")
        if data_root is not None:
            data_config["data_root"] = data_root

        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        return OTXDataModule(
            train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
            test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
            tile_config=TileConfig(**data_config.pop("tile_config", {})),
            **data_config,
        )

    @staticmethod
    def instantiate(
        config: dict,
        work_dir: PathLike | None = None,
        data_root: PathLike | None = None,
        **kwargs,
    ) -> tuple[Engine, dict[str, Any]]:
        """Instantiate an object from the configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
            work_dir (PathLike): Path to the working directory.
            data_root (PathLike): The root directory for data.

        Returns:
            tuple: A tuple containing the engine and the train kwargs dictionary.
        """
        datamodule = GetiConfigConverter.instantiate_datamodule(
            config=config,
            data_root=data_root,
            **kwargs,
        )

        # Update num_classes & Instantiate Model
        model_config = config.pop("model")
        model_config["init_args"]["label_info"] = datamodule.label_info
        model_config["init_args"]["data_input_params"] = DataInputParams(
            input_size=datamodule.input_size,
            mean=datamodule.input_mean,
            std=datamodule.input_std,
        ).as_dict()
        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(OTXModel, "model", required=False, fail_untyped=False, skip={"label_info"})
        model = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

        if hasattr(model, "tile_config"):
            model.tile_config = datamodule.tile_config

        # Instantiate Engine
        config_work_dir = config.pop("work_dir", config["engine"].pop("work_dir", None))
        config["engine"]["work_dir"] = work_dir if work_dir is not None else config_work_dir
        engine = create_engine(model=model, data=datamodule, **config["engine"])

        # Instantiate Engine.train Arguments
        engine_parser = ArgumentParser()
        train_arguments = engine_parser.add_method_arguments(
            engine.__class__,
            "train",
            skip={"accelerator", "devices"},
            fail_untyped=False,
        )
        # Update callbacks & logger dir as engine.work_dir
        if "callbacks" in config and config["callbacks"] is not None:
            for callback in config["callbacks"]:
                if "init_args" in callback and "dirpath" in callback["init_args"]:
                    callback["init_args"]["dirpath"] = engine.work_dir
        if "logger" in config and config["logger"] is not None:
            for logger in config["logger"]:
                if "save_dir" in logger["init_args"]:
                    logger["init_args"]["save_dir"] = engine.work_dir
                if "log_dir" in logger["init_args"]:
                    logger["init_args"]["log_dir"] = engine.work_dir
        instantiated_kwargs = engine_parser.instantiate_classes(Namespace(**config))

        train_kwargs = {k: v for k, v in instantiated_kwargs.items() if k in train_arguments}
        # enable auto batch size for training
        train_kwargs["adaptive_bs"] = "Safe"

        return engine, train_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Input ModelTemplate config")
    parser.add_argument("-i", "--data_root", help="Input dataset root path")
    parser.add_argument("-o", "--work_dir", help="Input work directory path")
    args = parser.parse_args()
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)
    otx_config = GetiConfigConverter.convert(config=config)
    engine, train_kwargs = GetiConfigConverter.instantiate(
        config=otx_config,
        data_root=args.data_root,
        work_dir=args.work_dir,
    )
    engine.train(**train_kwargs)
