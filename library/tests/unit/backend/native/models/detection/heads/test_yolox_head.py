# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of YOLOXHead.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_dense_heads/test_yolox_head.py
"""

import torch
from omegaconf import DictConfig

from otx.backend.native.models.detection.heads.yolox_head import YOLOXHeadModule, _fp32_argmax
from otx.backend.native.models.detection.utils.assigners import SimOTAAssigner
from otx.backend.native.models.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.backend.native.models.utils.utils import InstanceData


class TestYOLOXHeadModule:
    def test_predict_by_feat(self):
        s = 256
        img_metas = [
            {
                "img_shape": (s, s, 3),
                "scale_factor": (1.0, 1.0),
            },
        ]
        test_cfg = DictConfig({"score_thr": 0.01, "nms": {"type": "nms", "iou_threshold": 0.65}})
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg)
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)
        head.predict_by_feat(cls_scores, bbox_preds, objectnesses, img_metas, cfg=test_cfg, rescale=True, with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            rescale=False,
            with_nms=False,
        )

    def test_prepare_loss_inputs(self, mocker):
        s = 256
        img_metas = [
            {
                "img_shape": (s, s, 3),
                "scale_factor": 1,
            },
        ]
        train_cfg = {
            "assigner": SimOTAAssigner(center_radius=2.5),
        }
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, train_cfg=train_cfg)
        assert not head.use_l1
        assert isinstance(head.multi_level_cls_convs[0][0], Conv2dModule)

        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = [InstanceData(bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))]
        mocker.patch(
            "otx.backend.native.models.detection.heads.base_head.unpack_det_entity",
            return_value=(gt_instances, img_metas),
        )

        raw_dict = head.prepare_loss_inputs(x=feat, entity=mocker.MagicMock())
        for key in [
            "flatten_objectness",
            "flatten_cls_preds",
            "flatten_bbox_preds",
            "flatten_bboxes",
            "obj_targets",
            "cls_targets",
            "bbox_targets",
            "l1_targets",
            "num_total_samples",
            "num_pos",
            "pos_masks",
        ]:
            assert key in raw_dict

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=True, train_cfg=train_cfg)
        assert isinstance(head.multi_level_cls_convs[0][0], DepthwiseSeparableConvModule)


class TestFP32Argmax:
    """Tests for _fp32_argmax — NPU-compatible FP32 argmax replacement."""

    def test_matches_torch_argmax(self):
        """_fp32_argmax must produce the same result as torch.argmax for random inputs."""
        torch.manual_seed(42)
        scores = torch.randn(2, 100, 10)  # (B, K, C=10)
        expected = scores.argmax(dim=-1).float()
        result = _fp32_argmax(scores)
        assert result.dtype == torch.float32
        assert torch.equal(result, expected)

    def test_clear_winner_per_class(self):
        """When one class is clearly dominant, _fp32_argmax must pick it."""
        # Shape (1, 4, 3) — 4 boxes, 3 classes
        scores = torch.tensor([[[0.1, 0.9, 0.2], [0.8, 0.1, 0.3], [0.1, 0.2, 0.7], [0.5, 0.5, 0.6]]])
        result = _fp32_argmax(scores)
        assert result.shape == (1, 4)
        assert torch.equal(result, torch.tensor([[1.0, 0.0, 2.0, 2.0]]))

    def test_single_class(self):
        """With C=1, the argmax is always 0."""
        scores = torch.rand(3, 50, 1)
        result = _fp32_argmax(scores)
        assert torch.all(result == 0.0)

    def test_two_classes(self):
        """Binary classification case."""
        scores = torch.tensor([[[0.3, 0.7], [0.9, 0.1]]])
        result = _fp32_argmax(scores)
        assert torch.equal(result, torch.tensor([[1.0, 0.0]]))

    def test_tie_selects_lowest_index(self):
        """On a tie, _fp32_argmax should keep the first (lowest) index, matching torch.argmax."""
        scores = torch.tensor([[[0.5, 0.5, 0.5]]])
        result = _fp32_argmax(scores)
        assert result.item() == 0.0

    def test_many_classes(self):
        """Stress test with a large number of classes (simulates COCO-like scenarios)."""
        torch.manual_seed(123)
        scores = torch.randn(4, 200, 80)
        expected = scores.argmax(dim=-1).float()
        result = _fp32_argmax(scores)
        assert torch.equal(result, expected)


class TestYOLOXExportByFeatNPU:
    """Tests that export_by_feat (non-NMS path) produces correct labels using FP32 ops."""

    def test_export_by_feat_labels_match_reference(self):
        """Labels from export_by_feat (FP32 argmax) must match torch.argmax reference."""
        torch.manual_seed(0)
        num_classes = 4
        test_cfg = {"score_thr": 0.01, "nms": {"type": "nms", "iou_threshold": 0.65}, "max_per_img": 100}
        head = YOLOXHeadModule(
            num_classes=num_classes, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg
        )
        s = 256
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        dets, labels = head.export_by_feat(cls_scores, bbox_preds, objectnesses, cfg=test_cfg, with_nms=False)

        assert dets.shape == (1, 100, 5)
        assert labels.shape == (1, 100)
        assert labels.dtype == torch.float32

        # Verify labels are valid class indices
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)

    def test_export_by_feat_nonzero_labels(self):
        """With multiple classes, at least some labels should be non-zero (not all class 0)."""
        torch.manual_seed(7)
        num_classes = 5
        test_cfg = {"score_thr": 0.0, "nms": {"type": "nms", "iou_threshold": 0.65}, "max_per_img": 200}
        head = YOLOXHeadModule(
            num_classes=num_classes, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg
        )
        s = 128
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        _, labels = head.export_by_feat(cls_scores, bbox_preds, objectnesses, cfg=test_cfg, with_nms=False)

        # With random weights and 5 classes, not all labels should be 0
        unique_labels = labels.unique()
        assert len(unique_labels) > 1, f"All labels are the same value: {unique_labels}"
