# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMDet."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.detection.backbones.cspnext import CSPNeXtModule
from otx.backend.native.models.detection.heads.rtmdet_head import RTMDetSepBNHeadModule
from otx.backend.native.models.detection.necks.cspnext_pafpn import CSPNeXtPAFPNModule
from otx.backend.native.models.detection.rtmdet import RTMDet
from otx.data.entity.torch import OTXPredBatch


class TestRTMDet:
    def test_init(self) -> None:
        otx_rtmdet_tiny = RTMDet(
            model_name="rtmdet_tiny",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(otx_rtmdet_tiny.model.backbone, CSPNeXtModule)
        assert isinstance(otx_rtmdet_tiny.model.neck, CSPNeXtPAFPNModule)
        assert isinstance(otx_rtmdet_tiny.model.bbox_head, RTMDetSepBNHeadModule)
        assert otx_rtmdet_tiny.data_input_params.input_size == (640, 640)

    def test_exporter(self) -> None:
        otx_rtmdet_tiny = RTMDet(
            model_name="rtmdet_tiny",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        otx_rtmdet_tiny_exporter = otx_rtmdet_tiny._exporter
        assert isinstance(otx_rtmdet_tiny_exporter, OTXNativeModelExporter)
        assert otx_rtmdet_tiny_exporter.swap_rgb is True

    @pytest.mark.parametrize(
        "model",
        [
            RTMDet(
                model_name="rtmdet_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output

    @pytest.mark.parametrize(
        "model",
        [
            RTMDet(
                model_name="rtmdet_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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
            RTMDet(
                model_name="rtmdet_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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
            RTMDet(
                model_name="rtmdet_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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
