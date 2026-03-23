# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskRCNN architecture."""

import pytest
import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.instance_segmentation.maskrcnn import MaskRCNN
from otx.backend.native.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
from otx.data.entity.sample import OTXPredictionBatch
from otx.types.export import TaskLevelExportParameters


class TestMaskRCNN:
    @pytest.fixture(
        params=[
            ("MaskRCNN", "maskrcnn_resnet_50"),
            ("MaskRCNNTV", "maskrcnn_resnet_50"),
        ],
    )
    def fxt_model(self, request):
        cls_name, model_name = request.param
        cls = MaskRCNN if cls_name == "MaskRCNN" else MaskRCNNTV
        return cls(
            label_info=3,
            model_name=model_name,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_load_weights(self, mocker) -> None:
        model = MaskRCNN(
            label_info=2,
            model_name="maskrcnn_resnet_50",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)

    def test_loss(self, fxt_model, fxt_instance_seg_batch):
        data = fxt_instance_seg_batch

        output = fxt_model(data)
        if fxt_model.model_name == "maskrcnn_resnet_50" and isinstance(fxt_model, MaskRCNNTV):
            assert "loss_classifier" in output
            assert "loss_box_reg" in output
            assert "loss_mask" in output
            assert "loss_objectness" in output
            assert "loss_rpn_box_reg" in output
        else:
            assert "loss_cls" in output
            assert "loss_bbox" in output
            assert "loss_mask" in output
            assert "loss_cls_rpn" in output
            assert "loss_bbox_rpn" in output

    def test_predict(self, fxt_model, fxt_instance_seg_batch):
        data = fxt_instance_seg_batch
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        fxt_model.eval()
        output = fxt_model(data)
        assert isinstance(output, OTXPredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

        # TODO(Eugene): Explain should return proper output.
        # After enabling explain for maskrcnn, below codes shuold be passed
        # fxt_model.explain_mode = True  # noqa: ERA001
        # output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))  # noqa: ERA001
        # assert len(output) == 5  # noqa: ERA001
