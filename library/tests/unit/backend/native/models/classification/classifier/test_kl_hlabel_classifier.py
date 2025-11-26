# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from otx.backend.native.models.classification.backbones import EfficientNetBackbone
from otx.backend.native.models.classification.classifier import HLabelClassifier, KLHLabelClassifier
from otx.backend.native.models.classification.heads import LinearClsHead, MultiLabelLinearClsHead
from otx.backend.native.models.classification.heads.hlabel_cls_head import HierarchicalClsHead
from otx.backend.native.models.classification.losses import AsymmetricAngularLossWithIgnore
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling


class TestHierHead(HierarchicalClsHead):
    """Lightweight hierarchical head for tests, compatible with H/KLH classifiers."""

    def __init__(self, in_channels: int, head_class_sizes=(3, 3)):
        # e.g., two heads with 3 classes each -> total classes = 6
        self.head_class_sizes = list(head_class_sizes)
        num_multiclass_heads = len(self.head_class_sizes)
        num_multilabel_classes = 0
        num_single_label_classes = sum(self.head_class_sizes)
        num_classes = num_single_label_classes

        # Build per-head logit ranges, e.g. [(0,3), (3,6)]
        start = 0
        ranges = {}
        for idx, k in enumerate(self.head_class_sizes):
            ranges[str(idx)] = (start, start + k)
            start += k

        empty_multiclass_head_indices = []

        # Call the real base class with all required args
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            num_multiclass_heads=num_multiclass_heads,
            num_multilabel_classes=num_multilabel_classes,
            head_idx_to_logits_range=ranges,
            num_single_label_classes=num_single_label_classes,
            empty_multiclass_head_indices=empty_multiclass_head_indices,
        )

        # Simple linear head over pooled features -> logits
        self.classifier = nn.Linear(in_channels, num_classes)
        self._head_ranges = ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.classifier(x)


class TestKLHLabelClassifier:
    @pytest.fixture(
        params=[
            (LinearClsHead, nn.CrossEntropyLoss, "fxt_multiclass_cls_batch_data_entity"),
            (MultiLabelLinearClsHead, AsymmetricAngularLossWithIgnore, "fxt_multilabel_cls_batch_data_entity"),
        ],
        ids=["multiclass", "multilabel"],
    )
    def fxt_model_and_inputs(self, request):
        head_class_sizes = (3, 3)
        input_fxt_name = "fxt_multiclass_cls_batch_data_entity"
        backbone = EfficientNetBackbone(model_name="efficientnet_b0")
        neck = GlobalAveragePooling(dim=2)
        head = TestHierHead(in_channels=backbone.num_features, head_class_sizes=head_class_sizes)
        loss = nn.CrossEntropyLoss()
        fxt_input = request.getfixturevalue(input_fxt_name)
        level = len(head_class_sizes)
        fxt_labels = torch.stack(fxt_input.labels)
        fxt_labels = fxt_labels.repeat(1, level)
        return (backbone, neck, head, loss, fxt_input.images, fxt_labels)

    def test_forward(self, fxt_model_and_inputs):
        backbone, neck, head, loss, images, labels = fxt_model_and_inputs

        model = KLHLabelClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            multiclass_loss=loss,
            kl_weight=1,
        )

        output = model(images, labels, mode="loss")
        assert isinstance(output, torch.Tensor)

    def test_klh_loss_greater_than_hlabel(self, fxt_model_and_inputs):
        """KLHLabelClassifier should have strictly larger loss than HLabelClassifier
        when kl_weight > 0 and there are >= 2 multiclass heads."""
        backbone, neck, head, loss, images, labels = fxt_model_and_inputs
        h_model = HLabelClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            multiclass_loss=loss,
        )
        kl_h_model = KLHLabelClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            multiclass_loss=loss,
            kl_weight=2.0,
        )

        h_loss = h_model.loss(images, labels)
        klh_loss = kl_h_model.loss(images, labels)

        print(f"HLabel loss: {h_loss.item():.6f} | KLH loss: {klh_loss.item():.6f}")
        assert klh_loss > h_loss, "Expected KLH loss to be greater due to added KL term"

    def test_klh_weight_zero_match_hlabel(self, fxt_model_and_inputs):
        """With kl_weight == 0, KLH loss should match H label loss (within tolerance)."""
        backbone, neck, head, loss, images, labels = fxt_model_and_inputs
        h_model = HLabelClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            multiclass_loss=loss,
        )
        kl_h_model = KLHLabelClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
            multiclass_loss=loss,
            kl_weight=0,
        )
        h_loss = h_model.loss(images, labels)
        klh_loss = kl_h_model.loss(images, labels)

        print(f"[kl=0] HLabel loss: {h_loss.item():.6f} | KLH loss: {klh_loss.item():.6f}")
        assert torch.allclose(klh_loss, h_loss, atol=1e-6), "With kl_weight=0, losses should match"
