# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from getitune.backend.native.models.base import DataInputParams, OTXModel
from getitune.data.module import OTXDataModule
from getitune.tools.auto_configurator import (
    DEFAULT_CONFIG_PER_TASK,
    AutoConfigurator,
)
from getitune.types.label import LabelInfo, SegLabelInfo
from getitune.types.task import OTXTaskType
from getitune.utils.utils import should_pass_label_info


@pytest.fixture
def fxt_data_root_per_task_type() -> dict:
    return {
        OTXTaskType.MULTI_CLASS_CLS: "tests/assets/classification_cifar10",
        OTXTaskType.MULTI_LABEL_CLS: "tests/assets/multilabel_classification_coco",
        OTXTaskType.DETECTION: "tests/assets/detection_coco",
        OTXTaskType.KEYPOINT_DETECTION: "tests/assets/keypoint_detection_coco",
        OTXTaskType.ROTATED_DETECTION: "tests/assets/detection_coco",
        OTXTaskType.INSTANCE_SEGMENTATION: "tests/assets/instance_segmentation_coco",
        OTXTaskType.SEMANTIC_SEGMENTATION: "tests/assets/segmentation_pets",
    }


class TestAutoConfigurator:
    def test_check_task(self) -> None:
        # None inputs
        with pytest.raises(ValueError, match="Either task or model must be provided."):
            auto_configurator = AutoConfigurator(task=None, model=None)

        # data_root is None & task is not None
        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # instantiate with model_config_path
        model_config_path = "src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml"
        auto_configurator = AutoConfigurator(data_root=None, task=None, model=model_config_path)
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # instantiate with model_config_path
        with pytest.raises(
            ValueError,
            match="If model is provided as a name, task must be provided to find the model.",
        ):
            auto_configurator = AutoConfigurator(data_root=None, task=None, model="mobilenet_v3_large")

        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS", model="mobilenet_v3_large")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # data_root is not None & task is None
        data_root = "tests/assets/classification_cifar10"
        auto_configurator = AutoConfigurator(data_root=data_root, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

    def test_load_default_config(self) -> None:
        # Test the load_default_config function
        data_root = "tests/assets/classification_cifar10"
        task = OTXTaskType.MULTI_CLASS_CLS
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # Default Config
        default_config = auto_configurator._load_default_config()
        target_config = DEFAULT_CONFIG_PER_TASK[task].resolve()
        assert isinstance(default_config, dict)
        assert len(default_config) > 0
        assert "config" in default_config
        assert len(default_config["config"]) > 0
        assert str(default_config["config"][0]) == str(target_config)

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

        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        datamodule = auto_configurator.get_datamodule()
        assert isinstance(datamodule, OTXDataModule)
        assert datamodule.task == task

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
            data_input_params=DataInputParams((288, 288), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task="DETECTION")

        datamodule = auto_configurator.get_datamodule()
        # The detection base config has augmentations_cpu with Resize
        assert any("Resize" in aug.get("class_path", "") for aug in datamodule.test_subset.augmentations_cpu)

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")
        # OV recipes now use Resize (preprocessing moved from ModelAPI to OTX)
        assert len(updated_datamodule.test_subset.augmentations_cpu) == 1
        assert "Resize" in updated_datamodule.test_subset.augmentations_cpu[0]["class_path"]
        assert not updated_datamodule.tile_config.enable_tiler

    def test_update_ov_subset_pipeline_from_pre_constructed_datasets(self) -> None:
        """Test that update_ov_subset_pipeline works when the datamodule was created via from_otx_datasets (no data_root)."""
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=OTXTaskType.DETECTION)

        # Create a normal datamodule first, then rebuild it via from_otx_datasets
        # to simulate what the quantization pipeline does
        datamodule = auto_configurator.get_datamodule()
        pre_constructed_datamodule = OTXDataModule.from_otx_datasets(
            train_dataset=datamodule.subsets["train"],
            val_dataset=datamodule.subsets["val"],
            test_dataset=datamodule.subsets.get("test"),
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
        )
        assert pre_constructed_datamodule.data_root == ""

        # This should NOT raise ValueError about dataset format detection
        updated_datamodule = auto_configurator.update_ov_subset_pipeline(pre_constructed_datamodule, subset="train")
        assert len(updated_datamodule.train_subset.augmentations_cpu) == 1
        assert "Resize" in updated_datamodule.train_subset.augmentations_cpu[0]["class_path"]
        assert not updated_datamodule.tile_config.enable_tiler
        # Verify subsets are preserved
        assert "train" in updated_datamodule.subsets
        assert "val" in updated_datamodule.subsets
