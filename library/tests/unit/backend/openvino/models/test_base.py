# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the OpenVINO base model."""

import tempfile

import numpy as np
import openvino as ov
import pytest
import torch
from model_api.models.result import ClassificationResult
from pytest_mock import MockerFixture

from otx.backend.openvino.models import OVModel
from otx.data.entity.torch import OTXDataBatch


class TestOVModel:
    @pytest.fixture()
    def input_batch(self) -> OTXDataBatch:
        image = [torch.rand(3, 10, 10) for _ in range(3)]
        return OTXDataBatch(3, image, [])

    @pytest.fixture()
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
