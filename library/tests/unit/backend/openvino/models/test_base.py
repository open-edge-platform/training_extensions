# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the OpenVINO base model."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import openvino as ov
import pytest
import torch
from model_api.models.result import ClassificationResult

from getitune.backend.openvino.models import OVModel
from getitune.data.entity.sample import SampleBatch

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestOVModel:
    @pytest.fixture
    def input_batch(self) -> SampleBatch:
        image = [torch.rand(3, 10, 10) for _ in range(3)]
        return SampleBatch(images=torch.stack(image), labels=[])

    @pytest.fixture
    def model(self, get_dummy_ov_cls_model) -> OVModel:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ov.save_model(get_dummy_ov_cls_model, f"{tmp_dir}/model.xml")
            return OVModel(model_path=f"{tmp_dir}/model.xml", model_type="Classification")

    def test_create_model(self, model) -> None:
        pass

    def test_customize_inputs(self, model, input_batch) -> None:
        inputs = model._customize_inputs(input_batch)
        assert isinstance(inputs, dict)
        assert "inputs" in inputs
        assert inputs["inputs"][1].shape == np.transpose(input_batch.images[1].numpy(), (1, 2, 0)).shape

    def test_forward(self, model, input_batch, mocker: MockerFixture) -> None:
        model._customize_outputs = lambda x, _: x
        model.model.postprocess = mocker.Mock(return_value=ClassificationResult())
        outputs = model.forward(input_batch)
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert isinstance(outputs[2], ClassificationResult)

    def test_dummy_input(self, model: OVModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size


class TestResolveModelType:
    """Tests for OVModel.model_type metadata resolution."""

    @pytest.fixture
    def ov_model_instance(self, get_dummy_ov_cls_model) -> OVModel:
        """Create a minimal OVModel for testing model_type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ov.save_model(get_dummy_ov_cls_model, f"{tmp_dir}/model.xml")
            return OVModel(model_path=f"{tmp_dir}/model.xml", model_type="Classification")

    def test_returns_rt_info_model_type_when_present(self, ov_model_instance: OVModel) -> None:
        """When rt_info has model_type, model_type should return it."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "YOLO11"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "YOLO11"
        mock_adapter.model.has_rt_info.assert_called_once_with(["model_info", "model_type"])

    def test_returns_class_default_when_no_rt_info(self, ov_model_instance: OVModel) -> None:
        """When rt_info has no model_type, should fall back to constructor default."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = False

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_returns_class_default_when_rt_info_empty(self, ov_model_instance: OVModel) -> None:
        """When rt_info model_type is empty string, should fall back to constructor default."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = ""
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_returns_same_type_without_logging_when_matching(self, ov_model_instance: OVModel) -> None:
        """When rt_info model_type matches class default, should return it without override log."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "Classification"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_ssd_default_overridden_by_yolo11(self) -> None:
        """Simulates the OVDetectionModel case: default 'SSD' overridden by 'YOLO11' from rt_info."""
        ov_model = object.__new__(OVModel)
        ov_model._model_type = "SSD"
        ov_model.model_path = "model.xml"

        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "YOLO11"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model._model_adapter = mock_adapter
        result = ov_model.model_type
        assert result == "YOLO11"
