# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX RTMDetInst architecture."""

import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.instance_segmentation.rtmdet_inst import RTMDetInst
from otx.data.entity.torch import OTXPredBatch


class TestRTMDetInst:
    def test_loss(self, fxt_data_module):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 32, 32])
        data.masks = [torch.zeros((len(masks), 32, 32)) for masks in data.masks]

        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_mask" in output

    def test_predict(self, fxt_data_module):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, OTXPredBatch)

    def test_export(self):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

        # TODO(Eugene): Explain should return proper output.
        # After enabling explain for maskrcnn, below codes shuold be passed
        # model.explain_mode = True  # noqa: ERA001
        # output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))  # noqa: ERA001
        # assert len(output) == 5  # noqa: ERA001
