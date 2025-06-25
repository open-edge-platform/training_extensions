# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX YOLOX architecture."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.detection.backbones.csp_darknet import CSPDarknetModule
from otx.backend.native.models.detection.heads.yolox_head import YOLOXHeadModule
from otx.backend.native.models.detection.necks.yolox_pafpn import YOLOXPAFPNModule
from otx.backend.native.models.detection.yolox import YOLOX
from otx.data.entity.torch import OTXPredBatch


class TestYOLOX:
    def test_init(self) -> None:
        otx_yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(otx_yolox_l.model.backbone, CSPDarknetModule)
        assert isinstance(otx_yolox_l.model.neck, YOLOXPAFPNModule)
        assert isinstance(otx_yolox_l.model.bbox_head, YOLOXHeadModule)
        assert otx_yolox_l.data_input_params.input_size == (640, 640)

        otx_yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert otx_yolox_tiny.data_input_params.input_size == (640, 640)

        otx_yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((416, 416), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert otx_yolox_tiny.data_input_params.input_size == (416, 416)

    def test_exporter(self) -> None:
        otx_yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        otx_yolox_l_exporter = otx_yolox_l._exporter
        assert isinstance(otx_yolox_l_exporter, OTXNativeModelExporter)
        assert otx_yolox_l_exporter.swap_rgb is True

        otx_yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        otx_yolox_tiny_exporter = otx_yolox_tiny._exporter
        assert isinstance(otx_yolox_tiny_exporter, OTXNativeModelExporter)
        assert otx_yolox_tiny_exporter.swap_rgb is False

    @pytest.mark.parametrize(
        "model",
        [
            YOLOX(
                model_name="yolox_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_s",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_l",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_x",
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
        assert "loss_obj" in output

    @pytest.mark.parametrize(
        "model",
        [
            YOLOX(
                model_name="yolox_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_s",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_l",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_x",
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
            YOLOX(
                model_name="yolox_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_s",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_l",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_x",
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
            YOLOX(
                model_name="yolox_tiny",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_s",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_l",
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            YOLOX(
                model_name="yolox_x",
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
