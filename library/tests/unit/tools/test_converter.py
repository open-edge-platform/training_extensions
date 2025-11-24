# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pytest

from otx.tools.converter import GetiConfigConverter
from tests.integration.api.geti_otx_config_utils import OTXConfig


class TestGetiConfigConverter:
    def test_convert(self):
        otx_config = OTXConfig.from_yaml_file("tests/assets/geti/model_configs/detection.yaml")
        config = GetiConfigConverter.convert(asdict(otx_config))

        assert config["data"]["input_size"] == (992, 800)
        assert config["data"]["train_subset"]["batch_size"] == 4
        assert config["data"]["val_subset"]["batch_size"] == 4
        assert config["data"]["test_subset"]["batch_size"] == 8
        assert config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.001
        assert config["max_epochs"] == 100
        assert config["data"]["train_subset"]["num_workers"] == 2
        assert config["data"]["val_subset"]["num_workers"] == 2
        assert config["data"]["test_subset"]["num_workers"] == 2
        assert config["callbacks"][1]["init_args"]["patience"] == 10
        assert not config["data"]["tile_config"]["enable_tiler"]
        assert config["data"]["tile_config"]["overlap"] == 0.2
        assert config["data"]["tile_config"]["tile_size"] == (400, 400)

    def test_convert_task_overriding(self):
        otx_config = OTXConfig.from_yaml_file("tests/assets/geti/model_configs/classification.yaml")
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
        supported_augs_list_for_configuration = [
            "otx.data.transform_libs.torchvision.EfficientNetRandomCrop",
            "torchvision.transforms.v2.RandomPhotometricDistort",
            "otx.data.transform_libs.torchvision.RandomAffine",
            "otx.data.transform_libs.torchvision.RandomFlip",
            "torchvision.transforms.v2.RandomVerticalFlip",
            "otx.data.transform_libs.torchvision.RandomGaussianBlur",
            "otx.data.transform_libs.torchvision.RandomGaussianNoise",
        ]
        cfg_path = "tests/assets/geti/model_configs/classification.yaml"
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 9
        # default values
        list_of_all_augs = []
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] in [
                "otx.data.transform_libs.torchvision.RandomFlip",
                "otx.data.transform_libs.torchvision.EfficientNetRandomCrop",
            ]:
                assert aug["enable"]
            list_of_all_augs.append(aug["class_path"])
        # check if all supported augs are in the config
        for configuable_aug in supported_augs_list_for_configuration:
            assert configuable_aug in list_of_all_augs, f"{configuable_aug} is missing for configuration."
        # change config from geti to enable all augs
        for aug in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].values():
            aug["enable"] = True
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 9
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] in supported_augs_list_for_configuration:
                assert aug["enable"]

        # disable EfficientNetRandomCrop
        for aug_name, aug_conf in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].items():
            if aug_name == "random_resize_crop":
                aug_conf["enable"] = False
            # modify some aug for test
            if aug_name == "color_jitter":
                aug_conf["contrast"] = [0.0, 3.0]
                aug_conf["hue"] = [-0.1, 0.1]

        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 9
        found = False
        for aug in default_config["data"]["train_subset"]["transforms"]:
            assert aug["class_path"] != supported_augs_list_for_configuration[0]
            if aug["class_path"] == "otx.data.transform_libs.torchvision.Resize":
                found = True
            if aug["class_path"] == "torchvision.transforms.v2.RandomPhotometricDistort":
                assert aug["init_args"]["contrast"] == [0.0, 3.0]
                assert aug["init_args"]["hue"] == [-0.1, 0.1]
        assert found

        # instantiate
        data_root = "tests/assets/classification_dataset"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.transforms) == 9
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 9

    def test_detection_augs(self, tmp_path):
        supported_augs_list_for_configuration = [
            "otx.data.transform_libs.torchvision.MinIoURandomCrop",
            "torchvision.transforms.v2.RandomPhotometricDistort",
            "otx.data.transform_libs.torchvision.RandomAffine",
            "otx.data.transform_libs.torchvision.RandomFlip",
            "torchvision.transforms.v2.RandomVerticalFlip",
            "otx.data.transform_libs.torchvision.RandomGaussianBlur",
            "otx.data.transform_libs.torchvision.RandomGaussianNoise",
        ]
        cfg_path = "tests/assets/geti/model_configs/detection.yaml"
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 10
        # default values
        list_of_all_augs = []
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] in [
                "otx.data.transform_libs.torchvision.RandomFlip",
                "otx.data.transform_libs.torchvision.MinIoURandomCrop",
            ]:
                assert aug["enable"]
            list_of_all_augs.append(aug["class_path"])
        # check if all supported augs are in the config
        for configuable_aug in supported_augs_list_for_configuration:
            assert configuable_aug in list_of_all_augs, f"{configuable_aug} is missing for configuration."
        # change config from geti to enable all augs
        for aug in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].values():
            aug["enable"] = True
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 10
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] in supported_augs_list_for_configuration:
                assert aug["enable"]

        # disable iou_random_crop
        for aug_name, aug_conf in otx_config.hyper_parameters["dataset_preparation"]["augmentation"].items():
            if aug_name == "iou_random_crop":
                aug_conf["enable"] = False
            # modify some aug for test
            if aug_name == "color_jitter":
                aug_conf["contrast"] = [0.0, 3.0]
                aug_conf["hue"] = [-0.1, 0.1]
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 10
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] == "torchvision.transforms.v2.RandomPhotometricDistort":
                assert aug["init_args"]["contrast"] == [0.0, 3.0]
                assert aug["init_args"]["hue"] == [-0.1, 0.1]
                break

        # instantiate
        data_root = "tests/assets/car_tree_bug"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.transforms) == 10
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert (
            len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 9
        )  # 10 - disabled iou_random_crop

    def test_instance_seg_augs(self, tmp_path):
        cfg_path = "tests/assets/geti/model_configs/instance_segmentation.yaml"
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 10

        # instantiate
        data_root = "tests/assets/car_tree_bug"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.transforms) == 10
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        assert len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 5
        assert (  # Resize, ToDtype, Pad, Normalize
            len(engine.datamodule.val_dataloader().dataset.transforms.transforms) == 4
        )

    def test_semantic_segmentation_augs(self, tmp_path):
        cfg_path = "tests/assets/geti/model_configs/semantic_segmentation.yaml"
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 9
        # instantiate
        data_root = "tests/assets/common_semantic_segmentation_dataset"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.transforms) == 9
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        # crop, collor_jitter, flip, normalize, to_dtype
        assert len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 5

    def test_keypoint_detection_augs(self, tmp_path):
        cfg_path = "tests/assets/geti/model_configs/keypoint_detection.yaml"
        otx_config = OTXConfig.from_yaml_file(cfg_path)
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 6
        # instantiate
        data_root = "tests/assets/car_tree_bug_keypoint"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert len(engine.datamodule.train_subset.transforms) == 6
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        # top_down_affine, normalize, to_dtype
        assert len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 3
        # resize, pad, normalize, to_dtype
        assert len(engine.datamodule.val_dataloader().dataset.transforms.transforms) == 4

        otx_config.hyper_parameters["dataset_preparation"]["augmentation"]["topdown_affine"]["enable"] = False
        default_config = GetiConfigConverter.convert(asdict(otx_config))
        assert len(default_config["data"]["train_subset"]["transforms"]) == 6
        for aug in default_config["data"]["train_subset"]["transforms"]:
            if aug["class_path"] == "otx.data.transform_libs.torchvision.TopDownAffine":
                assert not aug["enable"]

        # instantiate
        data_root = "tests/assets/car_tree_bug_keypoint"
        engine, _ = GetiConfigConverter.instantiate(
            config=default_config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert engine.datamodule.train_dataloader().dataset.transforms is not None
        # top_down_affine, normalize, to_dtype
        assert len(engine.datamodule.train_dataloader().dataset.transforms.transforms) == 3
        # resize, normalize, to_dtype
        assert len(engine.datamodule.val_dataloader().dataset.transforms.transforms) == 3
        for aug in engine.datamodule.val_dataloader().dataset.transforms.transforms:
            assert aug.__class__.__name__ != "otx.data.transform_libs.torchvision.Pad"

    def test_instantiate(self, tmp_path):
        data_root = "tests/assets/car_tree_bug"
        otx_config = OTXConfig.from_yaml_file("tests/assets/geti/model_configs/detection.yaml")
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
        assert engine.datamodule.train_subset.num_workers == 2
        assert engine.datamodule.val_subset.num_workers == 2
        assert engine.datamodule.test_subset.num_workers == 2
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
