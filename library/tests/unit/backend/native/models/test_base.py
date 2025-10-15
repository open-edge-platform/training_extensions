# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from datumaro import LabelCategories
from datumaro.components.annotation import GroupType
from lightning import Trainer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from pytest_mock import MockerFixture

from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.backend.native.models.classification.hlabel_models.base import OTXHlabelClsModel
from otx.backend.native.models.classification.multiclass_models.base import OTXMulticlassClsModel
from otx.backend.native.models.segmentation.base import OTXSegmentationModel
from otx.backend.native.schedulers.warmup_schedulers import LinearWarmupScheduler
from otx.types.label import HLabelInfo, LabelInfo, SegLabelInfo


class MockNNModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(1, num_classes)
        self.head.weight.data = torch.arange(num_classes, dtype=torch.float32).reshape(num_classes, 1)
        self.head.bias.data = torch.arange(num_classes, dtype=torch.float32)


class TestOTXModel:
    def test_init(self, monkeypatch):
        monkeypatch.setattr(OTXModel, "input_size_multiplier", 10, raising=False)
        with pytest.raises(ValueError, match="Input size should be a multiple"):
            OTXModel(label_info=2, data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

    def test_training_step_none_loss(self, mocker: MockerFixture) -> None:
        mock_trainer = mocker.create_autospec(spec=Trainer)
        mock_trainer.world_size = 1
        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)) and mocker.patch.object(
            OTXModel,
            "forward",
            return_value=None,
        ):
            current_model = OTXModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            current_model.trainer = mock_trainer

        batch = {"input": torch.randn(2, 3)}
        batch_idx = 0

        with pytest.raises(ValueError, match="Loss is None."):
            current_model.training_step(batch, batch_idx)

    def test_smart_weight_loading(self, mocker) -> None:
        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(2)):
            prev_model = OTXModel(
                label_info=2,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            prev_model.label_info = ["car", "truck"]
            prev_state_dict = prev_model.state_dict()

        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            current_model = OTXModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            current_model.label_info = ["car", "bus", "truck"]
            mocker.patch.object(
                current_model,
                "_identify_classification_layers",
                return_value=["model.head.weight", "model.head.bias"],
            )
            current_model.load_state_dict_incrementally(
                {"state_dict": prev_state_dict, "hyper_parameters": {"label_info": prev_model.label_info}},
            )
            curr_state_dict = current_model.state_dict()

        indices = torch.Tensor([0, 2]).to(torch.int32)

        assert torch.allclose(curr_state_dict["model.backbone.weight"], prev_state_dict["model.backbone.weight"])
        assert torch.allclose(curr_state_dict["model.backbone.bias"], prev_state_dict["model.backbone.bias"])
        assert torch.allclose(
            curr_state_dict["model.head.weight"].index_select(0, indices),
            prev_state_dict["model.head.weight"],
        )
        assert torch.allclose(
            curr_state_dict["model.head.bias"].index_select(0, indices),
            prev_state_dict["model.head.bias"],
        )

    def test_label_info_dispatch(self, mocker):
        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            with pytest.raises(TypeError, match="invalid_label_info"):
                OTXModel(
                    label_info="invalid_label_info",
                    data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                )

            # Test with LabelInfo
            label_info = OTXModel(
                label_info=LabelInfo(
                    ["label_1", "label_2"],
                    label_ids=["1", "2"],
                    label_groups=[["label_1", "label_2"]],
                ),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            assert isinstance(label_info.label_info, LabelInfo)

            # Test with SegLabelInfo
            seg_label_info = OTXModel(
                label_info=SegLabelInfo.from_num_classes(3),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            assert isinstance(seg_label_info.label_info, SegLabelInfo)

        with mocker.patch.object(OTXMulticlassClsModel, "_create_model", return_value=MockNNModule(3)):
            # Test simple Classfication model loading checkpoint
            cls_model = OTXMulticlassClsModel(
                label_info=LabelInfo(
                    ["label_1", "label_2"],
                    label_ids=["1", "2"],
                    label_groups=[["label_1", "label_2"]],
                ),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            label_info_dict = {
                "label_ids": ["1", "2"],
                "label_names": ["label_1", "label_2"],
                "label_groups": [["label_1", "label_2"]],
            }
            cls_model.load_state_dict_incrementally(
                {"state_dict": cls_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
            )
            assert isinstance(cls_model.label_info, LabelInfo)
            # test if ignore_index is not set
            label_info_dict["ignore_index"] = 255
            with pytest.raises(TypeError, match=r"unexpected keyword argument.*ignore_index"):
                cls_model.load_state_dict_incrementally(
                    {"state_dict": cls_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
                )

        with mocker.patch.object(OTXSegmentationModel, "_create_model", return_value=MockNNModule(3)):
            # test segmentation model loading checkpoint with SegLabelInfo
            segmentation_model = OTXSegmentationModel(
                label_info=SegLabelInfo.from_num_classes(3),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                model_name="segmentation_model",
            )
            segmentation_model.load_state_dict_incrementally(
                {"state_dict": segmentation_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
            )
            assert isinstance(segmentation_model.label_info, SegLabelInfo)
            assert hasattr(segmentation_model.label_info, "ignore_index")
            assert segmentation_model.label_info.ignore_index == 255

        # test hlabel classification model loading checkpoint with HLabelInfo
        labels = [
            LabelCategories.Category(name="car", parent="vehicle"),
            LabelCategories.Category(name="truck", parent="vehicle"),
            LabelCategories.Category(name="plush toy", parent="plush toy"),
            LabelCategories.Category(name="No class"),
        ]
        label_groups = [
            LabelCategories.LabelGroup(
                name="Detection labels___vehicle",
                labels=["car", "truck"],
                group_type=GroupType.EXCLUSIVE,
            ),
            LabelCategories.LabelGroup(
                name="Detection labels___plush toy",
                labels=["plush toy"],
                group_type=GroupType.EXCLUSIVE,
            ),
            LabelCategories.LabelGroup(name="No class", labels=["No class"], group_type=GroupType.RESTRICTED),
        ]
        dm_label_categories = LabelCategories(items=labels, label_groups=label_groups)
        hlabel_info = HLabelInfo.from_dm_label_groups(dm_label_categories)
        hlabel_dict_label_info = hlabel_info.as_dict(normalize_label_names=True)

        with mocker.patch.object(OTXHlabelClsModel, "_create_model", return_value=MockNNModule(3)):
            hlabel_model = OTXHlabelClsModel(
                hlabel_dict_label_info,
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            hlabel_model.load_state_dict_incrementally(
                {"state_dict": hlabel_model.state_dict(), "hyper_parameters": {"label_info": hlabel_dict_label_info}},
            )

            with pytest.raises(TypeError, match=r"unexpected keyword argument.*num_multiclass_heads"):
                segmentation_model.load_state_dict_incrementally(
                    {
                        "state_dict": segmentation_model.state_dict(),
                        "hyper_parameters": {"label_info": hlabel_dict_label_info},
                    },
                )

            with pytest.raises(TypeError, match=r"unexpected keyword argument.*num_multiclass_heads"):
                cls_model.load_state_dict_incrementally(
                    {"state_dict": cls_model.state_dict(), "hyper_parameters": {"label_info": hlabel_dict_label_info}},
                )

    def test_lr_scheduler_step(self, mocker: MockerFixture) -> None:
        mock_linear_warmup_scheduler = mocker.create_autospec(spec=LinearWarmupScheduler)
        mock_main_scheduler = mocker.create_autospec(spec=torch.optim.lr_scheduler.LRScheduler)

        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            current_model = OTXModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )

        mock_trainer = mocker.create_autospec(spec=Trainer)
        mock_trainer.lr_scheduler_configs = [
            LRSchedulerConfig(mock_linear_warmup_scheduler),
            LRSchedulerConfig(mock_main_scheduler),
        ]
        current_model.trainer = mock_trainer

        # Assume that LinearWarmupScheduler is activated
        mock_linear_warmup_scheduler.activated = True
        for scheduler in [mock_linear_warmup_scheduler, mock_main_scheduler]:
            current_model.lr_scheduler_step(scheduler=scheduler, metric=None)

        # Assert mock_main_scheduler's step() is not called
        mock_main_scheduler.step.assert_not_called()

        mock_main_scheduler.reset_mock()

        # Assume that LinearWarmupScheduler is not activated
        mock_linear_warmup_scheduler.activated = False

        for scheduler in [mock_linear_warmup_scheduler, mock_main_scheduler]:
            current_model.lr_scheduler_step(scheduler=scheduler, metric=None)

        # Assert mock_main_scheduler's step() is called
        mock_main_scheduler.step.assert_called()

        # Regardless of the activation status, LinearWarmupScheduler can be called
        assert mock_linear_warmup_scheduler.step.call_count == 2

    def test_v1_checkpoint_loading(self, mocker):
        model = OTXModel(label_info=3, data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        mocker.patch.object(model, "load_from_otx_v1_ckpt", return_value={})
        mocker.patch.object(model, "_identify_classification_layers", return_value=[])
        v1_ckpt = {
            "model": {"state_dict": {"backbone": torch.randn(2, 2)}},
            "labels": {"label_0": (), "label_1": (), "label_2": ()},
            "VERSION": 1,
        }
        assert model.load_state_dict_incrementally(v1_ckpt) is None


class TestDataInputParams:
    def test_as_dict(self):
        params = DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        params_dict = params.as_dict()
        assert params_dict == {
            "input_size": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

    def test_as_ncwh(self):
        params = DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ncwh = params.as_ncwh(batch_size=4)
        assert ncwh == (4, 3, 224, 224)
