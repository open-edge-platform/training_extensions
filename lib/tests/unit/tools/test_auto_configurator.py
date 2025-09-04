# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch

from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.data.module import OTXDataModule
from otx.tools import auto_configurator as target_file
from otx.tools.auto_configurator import (
    DEFAULT_CONFIG_PER_TASK,
    AutoConfigurator,
)
from otx.types.label import LabelInfo, SegLabelInfo
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType
from otx.utils.utils import should_pass_label_info


@pytest.fixture()
def fxt_data_root_per_task_type() -> dict:
    return {
        OTXTaskType.MULTI_CLASS_CLS: "tests/assets/classification_dataset",
        OTXTaskType.MULTI_LABEL_CLS: "tests/assets/multilabel_classification",
        OTXTaskType.DETECTION: "tests/assets/car_tree_bug",
        OTXTaskType.ANOMALY: "tests/assets/anomaly_hazelnut",
        OTXTaskType.ANOMALY_CLASSIFICATION: "tests/assets/anomaly_hazelnut",
        OTXTaskType.ANOMALY_DETECTION: "tests/assets/anomaly_hazelnut",
        OTXTaskType.ANOMALY_SEGMENTATION: "tests/assets/anomaly_hazelnut",
        OTXTaskType.KEYPOINT_DETECTION: "tests/assets/car_tree_bug_keypoint",
        OTXTaskType.ROTATED_DETECTION: "tests/assets/car_tree_bug",
        OTXTaskType.INSTANCE_SEGMENTATION: "tests/assets/car_tree_bug",
        OTXTaskType.SEMANTIC_SEGMENTATION: "tests/assets/common_semantic_segmentation_dataset",
    }


class TestAutoConfigurator:
    def test_check_task(self) -> None:
        # None inputs
        with pytest.raises(ValueError, match="Either task or model_config_path must be provided."):
            auto_configurator = AutoConfigurator(task=None, model_config_path=None)

        # data_root is None & task is not None
        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # instantiate with model_config_path
        model_config_path = "src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml"
        auto_configurator = AutoConfigurator(data_root=None, task=None, model_config_path=model_config_path)
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # data_root is not None & task is None
        data_root = "tests/assets/classification_dataset"
        auto_configurator = AutoConfigurator(data_root=data_root, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

    def test_load_default_config(self) -> None:
        # Test the load_default_config function
        data_root = "tests/assets/classification_dataset"
        task = OTXTaskType.MULTI_CLASS_CLS
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # Default Config
        default_config = auto_configurator._load_default_config()
        target_config = DEFAULT_CONFIG_PER_TASK[task].resolve()
        assert isinstance(default_config, dict)
        assert len(default_config) > 0
        assert "config" in default_config
        assert len(default_config["config"]) > 0
        assert default_config["config"][0] == target_config

        # OTX-Mobilenet-v2
        # new_config
        model_name = "deit_tiny"
        new_config = auto_configurator._load_default_config(
            config_path="src/otx/recipe/classification/multi_class_cls/deit_tiny.yaml",
        )
        new_path = str(target_config).split("/")
        new_path[-1] = f"{model_name}.yaml"
        new_target_config = Path("/".join(new_path))
        assert isinstance(new_config, dict)
        assert len(new_config) > 0
        assert "config" in new_config
        assert len(new_config["config"]) > 0
        assert Path(new_config["config"][0]).name == new_target_config.name
        assert Path(new_config["config"][0]).exists()

    def test_get_datamodule(self) -> None:
        data_root = None
        task = OTXTaskType.DETECTION
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # data_root is None
        with pytest.raises(ValueError, match="No data root provided."):
            assert auto_configurator.get_datamodule() is None

        data_root = "tests/assets/car_tree_bug"
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        datamodule = auto_configurator.get_datamodule()
        assert isinstance(datamodule, OTXDataModule)
        assert datamodule.task == task

    def test_get_datamodule_set_input_size_multiplier(self, mocker) -> None:
        mock_otxdatamodule = mocker.patch.object(target_file, "OTXDataModule")
        auto_configurator = AutoConfigurator(
            data_root="tests/assets/car_tree_bug",
            task=OTXTaskType.DETECTION,
            model_config_path="src/otx/recipe/detection/yolox_tiny.yaml",
        )
        auto_configurator.config["data"]["input_size"] = "auto"

        auto_configurator.get_datamodule()

        assert mock_otxdatamodule.call_args.kwargs["input_size_multiplier"] == 32

    def test_get_model(self, fxt_task: OTXTaskType, fxt_data_root_per_task_type) -> None:
        if fxt_task is OTXTaskType.H_LABEL_CLS:
            pytest.xfail(reason="Not working")

        auto_configurator = AutoConfigurator(task=fxt_task, data_root=fxt_data_root_per_task_type[fxt_task])

        # With label_info
        label_names = ["class1", "class2", "class3"]
        label_info = (
            LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)
            if fxt_task != OTXTaskType.SEMANTIC_SEGMENTATION
            else SegLabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)
        )
        model = auto_configurator.get_model(
            label_info=label_info,
            data_input_params=DataInputParams((256, 256), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(model, OTXModel)

        model_cls = model.__class__

        if should_pass_label_info(model_cls):
            with pytest.raises(ValueError, match="Given model class (.*) requires a valid label_info to instantiate."):
                _ = auto_configurator.get_model(label_info=None)

    def test_get_model_set_input_size(self) -> None:
        auto_configurator = AutoConfigurator(task=OTXTaskType.MULTI_CLASS_CLS)
        label_names = ["class1", "class2", "class3"]
        label_info = LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)

        model = auto_configurator.get_model(
            label_info=label_info,
            data_input_params=DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        assert model.data_input_params.input_size == (300, 300)

    def test_update_ov_subset_pipeline(self) -> None:
        data_root = "tests/assets/car_tree_bug"
        auto_configurator = AutoConfigurator(data_root=data_root, task="DETECTION")

        datamodule = auto_configurator.get_datamodule()
        assert datamodule.test_subset.transforms == [
            {
                "class_path": "otx.data.transform_libs.torchvision.Resize",
                "init_args": {
                    "scale": (800, 992),
                },
            },
            {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": torch.float32}},
            {
                "class_path": "torchvision.transforms.v2.Normalize",
                "init_args": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]},
            },
        ]

        assert datamodule.test_subset.transform_lib_type == TransformLibType.TORCHVISION

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")
        assert updated_datamodule.test_subset.transforms == [{"class_path": "torchvision.transforms.v2.ToImage"}]

        assert updated_datamodule.test_subset.transform_lib_type == TransformLibType.TORCHVISION
        assert not updated_datamodule.tile_config.enable_tiler
