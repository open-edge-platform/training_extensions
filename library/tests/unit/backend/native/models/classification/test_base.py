# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification model module."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from torch import nn
from torch.optim import Optimizer

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.hlabel_models.base import OTXHlabelClsModel
from otx.backend.native.models.classification.multiclass_models.base import OTXMulticlassClsModel
from otx.backend.native.models.classification.multilabel_models.base import OTXMultilabelClsModel
from otx.types.export import TaskLevelExportParameters


class MockClsModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = nn.Sequential()
        self.head = nn.Linear(5, 2)

    def init_weights(self):
        pass


class TestOTXMulticlassClsModel:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        OTXMulticlassClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXMulticlassClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical
        assert model._export_parameters.output_raw_scores

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_class_cls_data_entity,
    ) -> None:
        model = OTXMulticlassClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_multi_class_cls_data_entity[1],
            fxt_multi_class_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input


class TestOTXMultilabelClsModel:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        OTXMultilabelClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
    ) -> None:
        model = OTXMultilabelClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_label_cls_data_entity,
    ) -> None:
        model = OTXMultilabelClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_multi_label_cls_data_entity[1],
            fxt_multi_label_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input


class TestOTXHlabelClsModel:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        OTXHlabelClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert model._export_parameters.hierarchical

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_h_label_cls_data_entity,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_h_label_cls_data_entity[1],
            fxt_h_label_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input

        model.label_info.num_multilabel_classes = 0
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_h_label_cls_data_entity[1],
            fxt_h_label_cls_data_entity[2],
        )
        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input

    def test_set_label_info(self, fxt_hlabel_multilabel_info):
        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert model.label_info.num_multilabel_classes == fxt_hlabel_multilabel_info.num_multilabel_classes

        fxt_hlabel_multilabel_info.num_multilabel_classes = 0
        model.label_info = fxt_hlabel_multilabel_info
        assert model.label_info.num_multilabel_classes == 0
