# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX RTMDetInst architecture."""

import onnxruntime as ort
import torch
from otx.algo.instance_segmentation.rtmdet_inst import RTMDetInst
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity
from otx.core.types.export import OTXExportFormatType


class TestRTMDetInst:
    def test_loss(self, fxt_data_module):
        model = RTMDetInst(3, "rtmdet_inst_tiny")
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 32, 32])
        data.masks = [torch.zeros((len(masks), 32, 32)) for masks in data.masks]

        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_mask" in output

    def test_predict(self, fxt_data_module):
        model = RTMDetInst(3, "rtmdet_inst_tiny")
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, InstanceSegBatchPredEntity)

    def test_export(self):
        model = RTMDetInst(3, "rtmdet_inst_tiny")
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

        # TODO(Eugene): Explain should return proper output.
        # After enabling explain for maskrcnn, below codes shuold be passed
        # model.explain_mode = True  # noqa: ERA001
        # output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))  # noqa: ERA001
        # assert len(output) == 5  # noqa: ERA001

    def test_onnx_export(self, tmp_path):
        model = RTMDetInst(3, "rtmdet_inst_tiny")
        model_path = model.export(
            output_dir=tmp_path,
            base_name="model",
            export_format=OTXExportFormatType.ONNX,
        )

        assert model_path.exists()

        session = ort.InferenceSession(
            model_path,
        )

        onnx_shapes = {}
        for onnx_input in session.get_inputs():
            onnx_shapes[onnx_input.name] = onnx_input.shape

        assert tuple(onnx_shapes["image"][2:]) == model.input_size, "Input shape mismatch"
