# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX YOLOX architecture."""

import pytest
import torch

from getitune.backend.native.exporter.native import OTXNativeModelExporter
from getitune.backend.native.models.base import DataInputParams
from getitune.backend.native.models.detection.backbones.csp_darknet import CSPDarknetModule
from getitune.backend.native.models.detection.heads.yolox_head import YOLOXHeadModule
from getitune.backend.native.models.detection.necks.yolox_pafpn import YOLOXPAFPNModule
from getitune.backend.native.models.detection.yolox import YOLOX
from getitune.data.entity.sample import OTXPredictionBatch


class TestYOLOX:
    @pytest.fixture(params=["yolox_tiny"])
    def fxt_model(self, request) -> YOLOX:
        return YOLOX(
            model_name=request.param,
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_init(self) -> None:
        otx_yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(otx_yolox_l.model.backbone, CSPDarknetModule)
        assert isinstance(otx_yolox_l.model.neck, YOLOXPAFPNModule)
        assert isinstance(otx_yolox_l.model.bbox_head, YOLOXHeadModule)
        assert otx_yolox_l.data_input_params.input_size == (320, 320)

        otx_yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert otx_yolox_tiny.data_input_params.input_size == (320, 320)

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
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        otx_yolox_l_exporter = otx_yolox_l._exporter
        assert isinstance(otx_yolox_l_exporter, OTXNativeModelExporter)
        assert otx_yolox_l_exporter.swap_rgb is True

        otx_yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        otx_yolox_tiny_exporter = otx_yolox_tiny._exporter
        assert isinstance(otx_yolox_tiny_exporter, OTXNativeModelExporter)
        assert otx_yolox_tiny_exporter.swap_rgb is False

    def test_loss(self, fxt_model, fxt_detection_batch):
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_obj" in output

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, OTXPredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
