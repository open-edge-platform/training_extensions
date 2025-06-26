# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from otx.backend.native.exporter.native import OTXModelExporter
from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.detection.atss import ATSS
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.data.entity.torch import OTXPredBatch
from otx.types.export import TaskLevelExportParameters


class TestATSS:
    def test(self, mocker) -> None:
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=2,
            data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, OTXModelExporter)

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(
                model_name="atss_mobilenetv2",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            ATSS(
                model_name="atss_resnext101",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_centerness" in output

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(
                model_name="atss_mobilenetv2",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            ATSS(
                model_name="atss_resnext101",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ],
    )
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, OTXPredBatch)

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(
                model_name="atss_mobilenetv2",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            ATSS(
                model_name="atss_resnext101",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ],
    )
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(
                model_name="atss_mobilenetv2",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            ATSS(
                model_name="atss_resnext101",
                label_info=3,
                data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ],
    )
    def test_compiled_model(self, model):
        # Set Compile Counter
        torch._dynamo.reset()
        cnt = CompileCounter()

        # Set model compile setting
        model.model = torch.compile(model.model, backend=cnt)

        # Prepare inputs
        x = torch.randn(1, 3, *model.data_input_params.input_size)
        model.model(x)
        assert cnt.frame_count == 1
