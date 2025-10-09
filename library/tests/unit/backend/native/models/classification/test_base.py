# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification model module."""

from __future__ import annotations

from types import MethodType
from unittest.mock import MagicMock
from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from torch import nn
from torch.optim import Optimizer

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.classifier import HLabelClassifier, KLHLabelClassifier
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

class TestOTXHlabelClsModelwithKL:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        OTXHlabelClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture()
    def model_instance(self, mock_optimizer, mock_scheduler, fxt_hlabel_multilabel_info):
        """Create a minimal instance of OTXHlabelClsModel for testing."""
        return OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

    def _prepare_instance_with_fake_create(self, model_instance):
        """Replace _create_model() with a dummy version that returns a sentinel object."""
        sentinel = object()

        def fake_create(self, head_config=None):
            return sentinel
        model_instance._create_model = MethodType(fake_create, model_instance)
        return sentinel

    def test_create_model_triggers_finalize_when_kl_positive(self, model_instance):
        """
        When kl_weight > 0, calling _create_model() should trigger _finalize_model().
        """
        model_instance.kl_weight = 1.0

        # Spy on _finalize_model before the first _create_model() call
        finalize_spy = MagicMock(side_effect=lambda m: m)
        model_instance._finalize_model = finalize_spy

        # Replace _create_model with a dummy implementation
        sentinel = self._prepare_instance_with_fake_create(model_instance)

        # Call _create_model(); wrapper logic should invoke _finalize_model internally
        out = model_instance._create_model()

        finalize_spy.assert_called_once_with(sentinel)
        assert out is sentinel

    def test_create_model_does_not_trigger_finalize_when_kl_zero(self, model_instance):
        """
        When kl_weight == 0, calling _create_model() should NOT trigger _finalize_model().
        """
        model_instance.kl_weight = 0.0

        finalize_spy = MagicMock(side_effect=lambda m: m)
        model_instance._finalize_model = finalize_spy

        sentinel = self._prepare_instance_with_fake_create(model_instance)
        out = model_instance._create_model()

        assert finalize_spy.call_count == 0, "_finalize_model() should not be triggered when kl_weight == 0"
        assert out is sentinel
