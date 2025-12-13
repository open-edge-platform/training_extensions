# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test of RFDETRDetector."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from otx.backend.native.models.detection.detectors.rfdetr import RFDETRDetector


class DummyLWDETR(nn.Module):
    """Dummy LWDETR model for testing."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._export_mode = False
        # Dummy encoder with backbone for explain mode testing
        self.encoder = DummyEncoder()

    def forward(self, x, targets=None):
        """Forward pass returning pred_boxes, pred_logits, pred_masks."""
        if isinstance(x, torch.Tensor):
            batch = x.shape[0]
        else:
            # Handle nested tensor
            batch = len(x.tensors) if hasattr(x, "tensors") else 2

        return {
            "pred_boxes": torch.rand(batch, 300, 4),
            "pred_logits": torch.rand(batch, 300, self.num_classes),
        }

    def export(self):
        """Enable export mode."""
        self._export_mode = True

    def __call__(self, x, targets=None):
        """Make callable for export mode."""
        if self._export_mode:
            # In export mode, return tuple format
            if isinstance(x, torch.Tensor):
                batch = x.shape[0]
            else:
                batch = 1
            return (
                torch.rand(batch, 300, 4),  # pred_boxes
                torch.rand(batch, 300, self.num_classes),  # pred_logits
                None,  # pred_masks
            )
        return self.forward(x, targets)


class DummyEncoder(nn.Module):
    """Dummy encoder with backbone for explain mode testing."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()


class DummyBackbone(nn.Module):
    """Dummy backbone for explain mode testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature map."""
        return self.conv(x)


class DummyCriterion(nn.Module):
    """Dummy criterion for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

    def forward(self, outputs, targets):
        """Return dummy losses."""
        return {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(0.5),
            "loss_giou": torch.tensor(0.3),
        }


class DummyPostprocessor:
    """Dummy postprocessor for testing."""

    def __call__(self, outputs, target_sizes):
        """Return dummy detection results."""
        batch = outputs["pred_logits"].shape[0]
        return [
            {
                "scores": torch.rand(100),
                "boxes": torch.rand(100, 4) * 640,
                "labels": torch.randint(0, 10, (100,)),
            }
            for _ in range(batch)
        ]


class TestRFDETRDetector:
    """Test class for RFDETRDetector."""

    @pytest.fixture
    def rfdetr_detector(self) -> RFDETRDetector:
        """Create RFDETRDetector instance for testing."""
        return RFDETRDetector(
            lwdetr_model=DummyLWDETR(num_classes=10),
            criterion=DummyCriterion(),
            postprocessor=DummyPostprocessor(),
            optimizer_configuration=None,
            input_size=560,
            multi_scale=False,
            group_detr=13,
        )

    @pytest.fixture
    def targets(self) -> list[dict]:
        """Create dummy targets for testing."""
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

    @pytest.fixture
    def images(self) -> torch.Tensor:
        """Create dummy images for testing."""
        return torch.randn(2, 3, 560, 560)

    def test_init(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test RFDETRDetector initialization."""
        assert rfdetr_detector.input_size == 560
        assert rfdetr_detector.multi_scale is False
        assert rfdetr_detector.group_detr == 13
        assert rfdetr_detector.lwdetr is not None
        assert rfdetr_detector.criterion is not None
        assert rfdetr_detector.postprocessor is not None

    def test_init_with_multi_scale(self) -> None:
        """Test RFDETRDetector initialization with multi-scale training."""
        detector = RFDETRDetector(
            lwdetr_model=DummyLWDETR(),
            criterion=DummyCriterion(),
            postprocessor=DummyPostprocessor(),
            input_size=560,
            multi_scale=True,
            group_detr=13,
        )
        assert detector.multi_scale is True
        assert len(detector.scales) > 0

    def test_generate_scales(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test scale generation for multi-scale training."""
        scales = rfdetr_detector._generate_scales(560)
        assert isinstance(scales, list)
        assert len(scales) > 0
        assert 560 in scales  # Base size should be in scales

    def test_postprocess(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test postprocess method."""
        outputs = {
            "pred_logits": torch.randn(2, 300, 10),
            "pred_boxes": torch.randn(2, 300, 4),
        }
        original_sizes = [(640, 640), (640, 640)]

        scores, boxes, labels = rfdetr_detector.postprocess(outputs, original_sizes)

        assert isinstance(scores, list)
        assert isinstance(boxes, list)
        assert isinstance(labels, list)
        assert len(scores) == 2
        assert len(boxes) == 2
        assert len(labels) == 2

    def test_export(self, rfdetr_detector: RFDETRDetector, images: torch.Tensor) -> None:
        """Test export method."""
        rfdetr_detector.eval()
        batch_img_metas = [{"img_shape": (560, 560)}, {"img_shape": (560, 560)}]

        result = rfdetr_detector.export(images, batch_img_metas, explain_mode=False)

        # Check that export mode was enabled on lwdetr
        assert rfdetr_detector.lwdetr._export_mode is True
        assert isinstance(result, tuple)
        assert len(result) == 3
        boxes, labels, scores = result
        assert boxes.shape[0] == 2
        assert labels.shape[0] == 2
        assert scores.shape[0] == 2

    def test_export_explain_mode(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test export method with explain mode."""
        rfdetr_detector.eval()

        # Set up explainability functions
        rfdetr_detector.feature_vector_fn = lambda feats: torch.ones(1, 128)
        rfdetr_detector.explain_fn = lambda logits: torch.ones(1, 10, 7, 7)

        images = torch.randn(1, 3, 560, 560)
        batch_img_metas = [{"img_shape": (560, 560)}]

        result = rfdetr_detector.export(images, batch_img_metas, explain_mode=True)

        assert isinstance(result, dict)
        assert "bboxes" in result
        assert "labels" in result
        assert "scores" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_optimizer_configuration(self) -> None:
        """Test that optimizer configuration is properly stored."""
        optimizer_config = [
            {"params": r"^(?=.*backbone).*$", "lr": 0.0001},
            {"params": r"^(?=.*decoder).*$", "lr": 0.001},
        ]

        detector = RFDETRDetector(
            lwdetr_model=DummyLWDETR(),
            criterion=DummyCriterion(),
            postprocessor=DummyPostprocessor(),
            optimizer_configuration=optimizer_config,
            input_size=560,
            multi_scale=False,
            group_detr=13,
        )

        assert detector.optimizer_configuration == optimizer_config

    def test_explainability_functions_default_none(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test that explainability functions are None by default."""
        assert rfdetr_detector.feature_vector_fn is None
        assert rfdetr_detector.explain_fn is None
