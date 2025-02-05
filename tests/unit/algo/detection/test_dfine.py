# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of D-Fine."""

from unittest.mock import MagicMock

import pytest
import torch
import torchvision

from otx.algo.detection.backbones.hgnetv2 import HGNetv2
from otx.algo.detection.d_fine import DFine
from otx.algo.detection.heads.dfine_decoder import DFINETransformer
from otx.algo.detection.losses.dfine_loss import DFINECriterion
from otx.algo.detection.necks.dfine_hybrid_encoder import HybridEncoder
from otx.algo.detection.rtdetr import DETR
from otx.core.data.entity.detection import DetBatchPredEntity


class TestDFine:
    @pytest.mark.parametrize(
        "model",
        [
            DFine(label_info=3, model_name="dfine_hgnetv2_x"),
        ],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 640, 640])
        model(data)

    @pytest.mark.parametrize(
        "model",
        [
            DFine(label_info=3, model_name="dfine_hgnetv2_x"),
        ],
    )
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, 640, 640)
        model.eval()
        output = model(data)
        assert isinstance(output, DetBatchPredEntity)

    @pytest.mark.parametrize(
        "model",
        [
            DFine(label_info=3, model_name="dfine_hgnetv2_x"),
        ],
    )
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 3

    @pytest.fixture()
    def dfine_model(self):
        num_classes = 10
        model_name = "dfine_hgnetv2_x"
        backbone = HGNetv2(model_name=model_name)
        encoder = HybridEncoder(model_name=model_name)
        decoder = DFINETransformer(
            model_name=model_name,
            num_classes=num_classes,
        )
        criterion = DFINECriterion(
            weight_dict={
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            alpha=0.75,
            gamma=2.0,
            reg_max=32,
            num_classes=num_classes,
        )
        return DETR(backbone=backbone, encoder=encoder, decoder=decoder, num_classes=10, criterion=criterion)

    @pytest.fixture()
    def targets(self):
        return [
            {
                "boxes": torch.tensor([[0.2739, 0.2848, 0.3239, 0.3348], [0.1652, 0.1109, 0.2152, 0.1609]]),
                "labels": torch.tensor([2, 2]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [0.6761, 0.8174, 0.7261, 0.8674],
                        [0.1652, 0.1109, 0.2152, 0.1609],
                        [0.2848, 0.9370, 0.3348, 0.9870],
                    ],
                ),
                "labels": torch.tensor([8, 2, 7]),
            },
        ]

    @pytest.fixture()
    def images(self):
        return torch.randn(2, 3, 640, 640)

    def test_dfine_forward(self, dfine_model, images, targets):
        dfine_model.train()
        output = dfine_model(images, targets)
        assert isinstance(output, dict)
        for key in output:
            assert key.startswith("loss_")
        assert "loss_bbox" in output
        assert "loss_vfl" in output
        assert "loss_giou" in output

    def test_dfine_postprocess(self, dfine_model):
        outputs = {
            "pred_logits": torch.randn(2, 100, 10),
            "pred_boxes": torch.randn(2, 100, 4),
        }
        original_sizes = [[640, 640], [640, 640]]
        result = dfine_model.postprocess(outputs, original_sizes)
        assert isinstance(result, tuple)
        assert len(result) == 3
        scores, boxes, labels = result
        assert isinstance(scores, list)
        assert isinstance(boxes, list)
        assert isinstance(boxes[0], torchvision.tv_tensors.BoundingBoxes)
        assert boxes[0].canvas_size == original_sizes[0]
        assert isinstance(labels, list)
        assert len(scores) == 2
        assert len(boxes) == 2
        assert len(labels) == 2

    def test_dfine_export(self, dfine_model, images):
        dfine_model.eval()
        dfine_model.num_top_queries = 10
        batch_img_metas = [{"img_shape": (740, 740), "scale_factor": 1.0}]
        result = dfine_model.export(images, batch_img_metas)
        assert isinstance(result, dict)
        assert "bboxes" in result
        assert "labels" in result
        assert "scores" in result
        assert result["bboxes"].shape == (2, 10, 4)
        # ensure no scaling
        assert torch.all(result["bboxes"] < 2)

    def test_set_input_size(self):
        input_size = 1280
        model = DETR(
            backbone=MagicMock(),
            encoder=MagicMock(),
            decoder=MagicMock(),
            num_classes=10,
            input_size=input_size,
        )

        expected_multi_scale = sorted([input_size - i * 32 for i in range(-5, 6)] + [input_size] * 2)

        assert sorted(model.multi_scale) == expected_multi_scale
