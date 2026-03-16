# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from pathlib import Path

import pytest

from otx.tools.converter import GetiConfigConverter
from tests.integration.api.geti_otx_config_utils import OTXConfig


def _class_paths(config: dict, subset: str = "train_subset") -> tuple[list[str], list[str]]:
    subset_cfg = config["data"][subset]
    cpu = [aug["class_path"] for aug in subset_cfg.get("augmentations_cpu", [])]
    gpu = [aug["class_path"] for aug in subset_cfg.get("augmentations_gpu", [])]
    return cpu, gpu


class TestGetiConfigConverter:
    def test_convert(self):
        otx_config = OTXConfig.from_yaml_file(Path("tests/assets/geti/model_configs/detection.yaml"))
        config = GetiConfigConverter.convert(asdict(otx_config))

        assert config["data"]["input_size"] == (992, 800)
        assert config["data"]["train_subset"]["batch_size"] == 4
        assert config["data"]["val_subset"]["batch_size"] == 4
        assert config["data"]["test_subset"]["batch_size"] == 8
        assert config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.001
        assert config["max_epochs"] == 100
        assert config["data"]["train_subset"]["num_workers"] == 4
        assert config["data"]["val_subset"]["num_workers"] == 4
        assert config["data"]["test_subset"]["num_workers"] == 4
        assert config["callbacks"][1]["init_args"]["patience"] == 10
        assert not config["data"]["tile_config"]["enable_tiler"]
        assert config["data"]["tile_config"]["overlap"] == 0.2
        assert config["data"]["tile_config"]["tile_size"] == (400, 400)

    def test_convert_task_overriding(self):
        otx_config = OTXConfig.from_yaml_file(Path("tests/assets/geti/model_configs/classification.yaml"))
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert default_config["task"] == "MULTI_CLASS_CLS"

        otx_config.sub_task_type = "MULTI_LABEL_CLS"
        override_config = GetiConfigConverter.convert(asdict(otx_config))
        assert override_config["task"] == "MULTI_LABEL_CLS"

        otx_config.sub_task_type = "H_LABEL_CLS"
        override_config = GetiConfigConverter.convert(asdict(otx_config))
        assert override_config["task"] == "H_LABEL_CLS"

        otx_config.sub_task_type = "DETECTION"
        with pytest.raises(FileNotFoundError):
            GetiConfigConverter.convert(asdict(otx_config))

    def test_classification_augs(self, tmp_path):
        supported_gpu_augs = [
            "kornia.augmentation.ColorJiggle",
            "kornia.augmentation.RandomAffine",
            "kornia.augmentation.RandomVerticalFlip",
            "kornia.augmentation.RandomGaussianBlur",
            "kornia.augmentation.RandomGaussianNoise",
        ]
        cfg_path = Path("tests/assets/geti/model_configs/classification.yaml")
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_paths, gpu_paths = _class_paths(default_config)
        assert cpu_paths == [
            "torchvision.transforms.v2.RandomResizedCrop",
        ]
        assert gpu_paths == [
            "kornia.augmentation.RandomHorizontalFlip",
            "kornia.augmentation.Normalize",
        ]

        # change config from geti to enable all augs
        for aug in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].values():  # pyrefly: ignore
            aug["enable"] = True
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_paths, gpu_paths = _class_paths(default_config)
        assert cpu_paths == [
            "torchvision.transforms.v2.RandomResizedCrop",
        ]
        for configurable_aug in supported_gpu_augs:
            assert configurable_aug in gpu_paths, f"{configurable_aug} is missing for configuration."

        # disable EfficientNetRandomCrop
        for aug_name, aug_conf in otx_config.hyper_parameters["dataset_preparation"][  # pyrefly: ignore
            "augmentation"
        ].items():
            if aug_name == "random_resize_crop":
                aug_conf["enable"] = False
            # modify some aug for test
            if aug_name == "color_jitter":
                aug_conf["contrast"] = [0.0, 3.0]
                aug_conf["hue"] = [-0.1, 0.1]

        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_augs = default_config["data"]["train_subset"]["augmentations_cpu"]
        gpu_augs = default_config["data"]["train_subset"]["augmentations_gpu"]
        assert cpu_augs[0]["class_path"] == "otx.data.augmentation.transforms.Resize"
        color_jitter = next(aug for aug in gpu_augs if aug["class_path"] == "kornia.augmentation.ColorJiggle")
        assert color_jitter["init_args"]["contrast"] == [0.0, 3.0]
        assert color_jitter["init_args"]["hue"] == [-0.1, 0.1]

        # instantiate
        data_root = "tests/assets/classification_dataset"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )

        assert len(engine.datamodule.train_subset.augmentations_cpu) == 1  # Resize
        assert len(engine.datamodule.train_subset.augmentations_gpu) == 7
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert (
            len(engine.datamodule.train_dataloader().dataset.transforms.augmentations) == 2
        )  # +1 for intensity transform

    def test_detection_augs(self, tmp_path):
        supported_gpu_augs = [
            "kornia.augmentation.RandomAffine",
            "kornia.augmentation.RandomVerticalFlip",
            "kornia.augmentation.RandomGaussianBlur",
            "kornia.augmentation.RandomGaussianNoise",
        ]
        cfg_path = Path("tests/assets/geti/model_configs/detection.yaml")
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_paths, gpu_paths = _class_paths(default_config)
        assert cpu_paths == [
            "torchvision.transforms.v2.RandomIoUCrop",
            "torchvision.transforms.v2.SanitizeBoundingBoxes",
            "otx.data.augmentation.transforms.Resize",
        ]
        assert gpu_paths == [
            "kornia.augmentation.RandomHorizontalFlip",
            "kornia.augmentation.Normalize",
        ]

        # change config from geti to enable all augs
        for aug in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].values():  # pyrefly: ignore
            aug["enable"] = True
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_paths, gpu_paths = _class_paths(default_config)
        assert cpu_paths == [
            "torchvision.transforms.v2.RandomIoUCrop",
            "otx.data.augmentation.transforms.Resize",
        ]
        for configurable_aug in supported_gpu_augs:
            assert configurable_aug in gpu_paths, f"{configurable_aug} is missing for configuration."

        # disable iou_random_crop
        for aug_name, aug_conf in otx_config.hyper_parameters["dataset_preparation"][  # pyrefly: ignore
            "augmentation"
        ].items():
            if aug_name == "iou_random_crop":
                aug_conf["enable"] = False
            # modify some aug for test
            if aug_name == "color_jitter":
                aug_conf["contrast"] = [0.0, 3.0]
                aug_conf["hue"] = [-0.1, 0.1]
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_augs = default_config["data"]["train_subset"]["augmentations_cpu"]
        gpu_augs = default_config["data"]["train_subset"]["augmentations_gpu"]
        assert [aug["class_path"] for aug in cpu_augs] == [
            "otx.data.augmentation.transforms.Resize",
        ]
        color_jitter = next(aug for aug in gpu_augs if aug["class_path"] == "kornia.augmentation.ColorJiggle")
        assert color_jitter["init_args"]["contrast"] == [0.0, 3.0]
        assert color_jitter["init_args"]["hue"] == [-0.1, 0.1]

        # instantiate
        data_root = "tests/assets/car_tree_bug"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.augmentations_cpu) == 1
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert (
            len(engine.datamodule.train_dataloader().dataset.transforms.augmentations) == 2
        )  # +1 for intensity transform

    def test_instance_seg_augs(self, tmp_path):
        cfg_path = Path("tests/assets/geti/model_configs/instance_segmentation.yaml")
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        cpu_paths, gpu_paths = _class_paths(default_config)
        assert cpu_paths == ["otx.data.augmentation.transforms.Resize"]
        assert gpu_paths == [
            "kornia.augmentation.RandomHorizontalFlip",
            "kornia.augmentation.Normalize",
        ]

        # instantiate
        data_root = "tests/assets/car_tree_bug"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.augmentations_cpu) == 1
        assert len(engine.datamodule.train_subset.augmentations_gpu) == 2
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert (
            len(engine.datamodule.train_dataloader().dataset.transforms.augmentations) == 2
        )  # +1 for intensity transform
        assert len(engine.datamodule.val_dataloader().dataset.transforms.augmentations) == 2

    def test_instantiate(self, tmp_path):
        data_root = "tests/assets/car_tree_bug"
        otx_config = OTXConfig.from_yaml_file(Path("tests/assets/geti/model_configs/detection.yaml"))
        config = GetiConfigConverter.convert(asdict(otx_config))
        engine, train_kwargs = GetiConfigConverter.instantiate(
            config=config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert engine.work_dir == tmp_path

        assert engine.datamodule.data_root == data_root
        assert engine.datamodule.train_subset.batch_size == 4
        assert engine.datamodule.val_subset.batch_size == 4
        assert engine.datamodule.test_subset.batch_size == 8
        assert engine.datamodule.train_subset.num_workers == 4
        assert engine.datamodule.val_subset.num_workers == 4
        assert engine.datamodule.test_subset.num_workers == 4
        assert not engine.datamodule.tile_config.enable_tiler
        assert engine.datamodule.tile_config.enable_adaptive_tiling
        assert engine.datamodule.input_size == (992, 800)
        assert engine.model.data_input_params.input_size == (992, 800)

        assert len(train_kwargs["callbacks"]) == len(config["callbacks"])
        assert train_kwargs["callbacks"][1].patience == 10
        if "logger" in train_kwargs and train_kwargs["logger"] is not None:
            assert len(train_kwargs["logger"]) == len(config["logger"])
        assert train_kwargs["max_epochs"] == 100
        assert "adaptive_bs" in train_kwargs
        assert train_kwargs["adaptive_bs"] == "Safe"
