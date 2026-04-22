# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for instance segmentation model entity."""

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.backend.lightning.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
from getitune.backend.lightning.tools.explain.explain_algo import feature_vector_fn
from getitune.types.export import TaskLevelExportParameters


class TestLightningInstanceSegModel:
    @pytest.fixture
    def model(self) -> LightningInstanceSegModel:
        return MaskRCNNTV(
            label_info=1,
            model_name="maskrcnn_resnet_50",
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_create_model(self, model) -> None:
        mmdet_model = model._create_model()
        assert mmdet_model is not None
        assert isinstance(mmdet_model, torch.nn.Module)

    def test_get_explain_fn(self, model):
        model.explain_mode = True
        explain_fn = model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_inst_seg(self, model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = torch.randn(1, 3, 224, 224)
        model.model.feature_vector_fn = feature_vector_fn
        model.model.explain_fn = model.get_explain_fn()
        model.eval()
        result = model._forward_explain_inst_seg(model.model, inputs, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, model, fxt_inst_seg_data_entity) -> None:
        output_data = model._customize_inputs(fxt_inst_seg_data_entity[2])
        assert output_data["entity"] == fxt_inst_seg_data_entity[2]

    def test_forward_explain(self, model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = [image.float() for image in inputs.images]
        model.training = False
        model.eval()
        model.explain_mode = True
        outputs = model.forward_explain(inputs)

        assert outputs.saliency_map is not None
        assert len(outputs.saliency_map) > 0
        assert outputs.feature_vector is not None
        assert outputs.saliency_map is not None

    def test_reset_restore_model_forward(self, model):
        model.explain_mode = True
        initial_model_forward = model.model.forward

        model._reset_model_forward()
        assert model.original_model_forward is not None
        assert str(model.model.forward) != str(model.original_model_forward)

        model._restore_model_forward()
        assert model.original_model_forward is None
        assert str(model.model.forward) == str(initial_model_forward)

    def test_export_parameters(self, model):
        parameters = model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "instance_segmentation"

    def test_dummy_input(self, model):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
