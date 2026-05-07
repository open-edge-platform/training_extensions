# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune YOLOX architecture."""

import pytest
import torch

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.backbones.csp_darknet import CSPDarknetModule
from getitune.backend.lightning.models.detection.heads.yolox_head import YOLOXHeadModule
from getitune.backend.lightning.models.detection.necks.yolox_pafpn import YOLOXPAFPNModule
from getitune.backend.lightning.models.detection.yolox import YOLOX
from getitune.data.entity.sample import PredictionBatch


class TestYOLOX:
    @pytest.fixture(params=["yolox_tiny"])
    def fxt_model(self, request) -> YOLOX:
        return YOLOX(
            model_name=request.param,
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_init(self) -> None:
        yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(yolox_l.model.backbone, CSPDarknetModule)
        assert isinstance(yolox_l.model.neck, YOLOXPAFPNModule)
        assert isinstance(yolox_l.model.bbox_head, YOLOXHeadModule)
        assert yolox_l.data_input_params.input_size == (320, 320)

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert yolox_tiny.data_input_params.input_size == (320, 320)

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((416, 416), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert yolox_tiny.data_input_params.input_size == (416, 416)

    def test_exporter(self) -> None:
        yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        yolox_l_exporter = yolox_l._exporter
        assert isinstance(yolox_l_exporter, LightningModelExporter)
        assert yolox_l_exporter.swap_rgb is True

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        yolox_tiny_exporter = yolox_tiny._exporter
        assert isinstance(yolox_tiny_exporter, LightningModelExporter)
        assert yolox_tiny_exporter.swap_rgb is False

    def test_loss(self, fxt_model, fxt_detection_batch):
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_obj" in output

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4

    def test_export_without_nms(self, fxt_model):
        fxt_model.eval()
        fxt_model.export_nms = False
        dets, labels = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert dets.ndim == 3
        assert dets.shape[0] == 1
        assert dets.shape[2] == 5
        assert labels.shape == dets.shape[:2]

    @pytest.mark.parametrize(
        ("model_name", "expect_cleared"),
        [("yolox_s", True), ("yolox_l", True), ("yolox_x", True), ("yolox_tiny", False)],
    )
    def test_intensity_config_cleared_for_raw_uint8_models(self, model_name, expect_cleared):
        from getitune.config.data import IntensityConfig

        intensity_cfg = IntensityConfig(mode="scale_to_unit", max_value=255.0)
        params = DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), intensity_config=intensity_cfg)
        model = YOLOX(model_name=model_name, label_info=3, data_input_params=params)
        if expect_cleared:
            assert model.data_input_params.intensity_config is None
        else:
            assert model.data_input_params.intensity_config is not None
            assert model.data_input_params.intensity_config.mode == "scale_to_unit"

    @pytest.mark.parametrize("model_name", ["yolox_s", "yolox_l", "yolox_x"])
    @pytest.mark.parametrize("storage_dtype", ["uint16", "int16"])
    def test_raw_uint8_models_reject_high_bit_depth(self, model_name, storage_dtype):
        from getitune.config.data import IntensityConfig

        intensity_cfg = IntensityConfig(storage_dtype=storage_dtype, mode="scale_to_unit")
        params = DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), intensity_config=intensity_cfg)
        with pytest.raises(ValueError, match="does not support high-bit-depth"):
            YOLOX(model_name=model_name, label_info=3, data_input_params=params)
