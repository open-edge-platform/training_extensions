# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RF-DETR detection model."""

from __future__ import annotations

import pytest
import torch

from otx.backend.native.models.detection.rfdetr import RFDETR
from otx.data.entity.torch import OTXPredBatch


class TestRFDETR:
    """Test class for RF-DETR detection model."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
            "rfdetr_small",
            "rfdetr_base",
            "rfdetr_medium",
            "rfdetr_large",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test RF-DETR model initialization."""
        model = RFDETR(
            model_name=model_name,
            label_info=3,
        )
        assert model.model_name == model_name
        assert model.num_classes == 3

    def test_create_model(self) -> None:
        """Test RF-DETR model creation."""
        model = RFDETR(
            model_name="rfdetr_base",
            label_info=10,
        )
        created_model = model._create_model()
        assert created_model is not None
        assert isinstance(created_model, torch.nn.Module)

        # Check if the model has the expected components
        assert hasattr(created_model, "lwdetr")
        assert hasattr(created_model, "criterion")
        assert hasattr(created_model, "postprocessor")

    def test_default_preprocessing_params(self) -> None:
        """Test default preprocessing parameters for different model variants."""
        model = RFDETR(
            model_name="rfdetr_base",
            label_info=3,
        )

        # Check that default params use 0-255 range normalization
        default_params = model._default_preprocessing_params
        assert "rfdetr_base" in default_params
        assert default_params["rfdetr_base"].input_size == (560, 560)
        # ImageNet mean in 0-255 range
        assert default_params["rfdetr_base"].mean == (123.675, 116.28, 103.53)
        assert default_params["rfdetr_base"].std == (58.395, 57.12, 57.375)

    def test_optimizer_configuration(self) -> None:
        """Test that optimizer configuration is properly set."""
        model = RFDETR(
            model_name="rfdetr_base",
            label_info=5,
        )
        created_model = model._create_model()

        # Check optimizer configuration exists
        assert hasattr(created_model, "optimizer_configuration")
        assert created_model.optimizer_configuration is not None
        assert len(created_model.optimizer_configuration) > 0

    @pytest.mark.parametrize(
        ("model_name", "label_info"),
        [
            ("rfdetr_nano", 3),
            ("rfdetr_small", 5),
            ("rfdetr_base", 10),
        ],
    )
    def test_loss_computation(self, model_name: str, label_info: int, fxt_data_module) -> None:
        """Test RF-DETR loss computation in training mode."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_base": (560, 560),
        }
        input_size = input_sizes[model_name]

        model = RFDETR(
            model_name=model_name,
            label_info=label_info,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Get data batch
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, *input_size)

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(data)

        # Check that output contains loss components
        assert isinstance(output, dict)

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
            "rfdetr_small",
            "rfdetr_base",
        ],
    )
    def test_predict(self, model_name: str, fxt_data_module) -> None:
        """Test RF-DETR prediction in evaluation mode."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_base": (560, 560),
        }
        input_size = input_sizes[model_name]

        model = RFDETR(
            model_name=model_name,
            label_info=3,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Get data batch
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, *input_size)

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(data)

        # Check that output is OTXPredBatch
        assert isinstance(output, OTXPredBatch)
        assert output.batch_size == 2

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
            "rfdetr_base",
        ],
    )
    def test_export(self, model_name: str) -> None:
        """Test RF-DETR export functionality."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_base": (560, 560),
        }
        input_size = input_sizes[model_name]

        model = RFDETR(
            model_name=model_name,
            label_info=3,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, *input_size))
        assert len(output) == 3  # Should return boxes, labels, scores

    def test_export_explain_mode(self) -> None:
        """Test RF-DETR export with explain mode."""
        model = RFDETR(
            model_name="rfdetr_base",
            label_info=3,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Set model to evaluation mode with explain mode
        model.eval()
        model.explain_mode = True

        # Test export forward pass with explain mode
        output = model.forward_for_tracing(torch.randn(1, 3, 560, 560))
        assert isinstance(output, dict)
        assert "bboxes" in output
        assert "labels" in output
        assert "scores" in output
        assert "feature_vector" in output
        assert "saliency_map" in output

    def test_multi_scale_training(self) -> None:
        """Test RF-DETR with multi-scale training enabled."""
        model = RFDETR(
            model_name="rfdetr_nano",  # Use smaller model for faster test
            label_info=3,
            multi_scale=True,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Multi-scale should be enabled in the detector
        assert model.model.multi_scale is True
        assert isinstance(model.model.scales, list)
        assert len(model.model.scales) > 0
