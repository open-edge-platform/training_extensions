# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMDet."""

import pytest
import torch

from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.detection.backbones.cspnext import CSPNeXtModule
from otx.backend.native.models.detection.heads.rtmdet_head import RTMDetSepBNHeadModule
from otx.backend.native.models.detection.necks.cspnext_pafpn import CSPNeXtPAFPNModule
from otx.backend.native.models.detection.rtmdet import RTMDet
from otx.data.entity.sample import OTXPredictionBatch


class TestRTMDet:
    @pytest.fixture
    def fxt_model(self) -> RTMDet:
        return RTMDet(
            model_name="rtmdet_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_init(self, fxt_model) -> None:
        assert isinstance(fxt_model.model.backbone, CSPNeXtModule)
        assert isinstance(fxt_model.model.neck, CSPNeXtPAFPNModule)
        assert isinstance(fxt_model.model.bbox_head, RTMDetSepBNHeadModule)
        assert fxt_model.data_input_params.input_size == (320, 320)

    def test_exporter(self, fxt_model) -> None:
        exporter = fxt_model._exporter
        assert isinstance(exporter, OTXNativeModelExporter)
        assert exporter.swap_rgb is True

    def test_loss(self, fxt_model, fxt_detection_batch):
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output

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
