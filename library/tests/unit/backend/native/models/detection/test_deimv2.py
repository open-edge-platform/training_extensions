# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIMV2 detection model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from getitune.backend.native.models.base import DataInputParams
from getitune.backend.native.models.detection.deimv2 import DEIMV2
from getitune.data.entity.sample import OTXPredictionBatch


class TestDEIMV2:
    """Test class for DEIMV2 detection model."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
            "deimv2_m",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test DEIMV2 model initialization."""
        model = DEIMV2(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert model.model_name == model_name
        assert model.num_classes == 3
        assert model.data_input_params.input_size == (640, 640)
        assert model.input_size_multiplier == 32
        assert model_name in model._pretrained_weights

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_create_model(self, mock_load_checkpoint: MagicMock) -> None:
        """Test DEIMV2 model creation."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=10,
        )
        created_model = model._create_model()
        assert created_model is not None
        assert isinstance(created_model, torch.nn.Module)

        # Check if the model has the expected components
        assert hasattr(created_model, "backbone")
        assert hasattr(created_model, "encoder")
        assert hasattr(created_model, "decoder")
        assert hasattr(created_model, "criterion")
        assert hasattr(created_model, "num_classes")
        assert created_model.num_classes == 10

        # Verify load_checkpoint was called (may be called multiple times for backbone and model)
        assert mock_load_checkpoint.call_count >= 1

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_backbone_lr_mapping(self, mock_load_checkpoint: MagicMock) -> None:
        """Test that backbone learning rate mapping works correctly."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        created_model = model._create_model()

        # Check optimizer configuration exists
        assert hasattr(created_model, "optimizer_configuration")
        assert len(created_model.optimizer_configuration) == 3

    @pytest.mark.parametrize(
        ("model_name", "expected_lr"),
        [
            ("deimv2_x", 0.00001),
            ("deimv2_s", 0.000025),
        ],
    )
    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_backbone_lr_values(self, mock_load_checkpoint: MagicMock, model_name: str, expected_lr: float) -> None:
        """Test that backbone learning rates are correctly set for each model variant."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name=model_name,
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        created_model = model._create_model()

        # Check that the first optimizer config has the expected backbone lr
        assert created_model.optimizer_configuration[0]["lr"] == expected_lr

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_loss_computation(self, mock_load_checkpoint: MagicMock, fxt_detection_batch) -> None:
        """Test DEIMV2 loss computation in training mode."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=10,
        )

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(fxt_detection_batch)

        # Check that output contains expected DEIM loss components
        assert isinstance(output, dict)
        expected_losses = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_mal"]

        for loss_name in expected_losses:
            assert loss_name in output
            assert isinstance(output[loss_name], torch.Tensor)

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
        ],
    )
    def test_predict(self, mock_load_checkpoint: MagicMock, model_name: str, fxt_detection_batch) -> None:
        """Test DEIMV2 prediction in evaluation mode."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(fxt_detection_batch)

        # Check that output is OTXPredictionBatch
        assert isinstance(output, OTXPredictionBatch)
        assert output.batch_size == 2

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
        ],
    )
    def test_export(self, mock_load_checkpoint: MagicMock, model_name: str) -> None:
        """Test DEIMV2 export functionality."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name=model_name,
            label_info=3,
        )

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 3  # Should return boxes, scores, labels

        # Test with explain mode
        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 5  # Should return boxes, scores, labels, saliency_map, feature_vector

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_dinov3_backbone(self, mock_load_checkpoint: MagicMock) -> None:
        """Test that DEIMV2 uses DINOv3STA backbone."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()

        # Check that backbone is DINOv3STAsModule
        from getitune.backend.native.models.detection.backbones.dinov3sta import DINOv3STAsModule

        assert isinstance(created_model.backbone, DINOv3STAsModule)

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_hybrid_encoder(self, mock_load_checkpoint: MagicMock) -> None:
        """Test that DEIMV2 uses HybridEncoder."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()

        # Check that encoder is HybridEncoderModule
        from getitune.backend.native.models.detection.necks.dfine_hybrid_encoder import HybridEncoderModule

        assert isinstance(created_model.encoder, HybridEncoderModule)

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_deim_transformer_decoder(self, mock_load_checkpoint: MagicMock) -> None:
        """Test that DEIMV2 uses DEIMTransformer decoder."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()

        # Check that decoder is DEIMTransformerModule
        from getitune.backend.native.models.detection.heads.deim_decoder import DEIMTransformerModule

        assert isinstance(created_model.decoder, DEIMTransformerModule)

    @patch("getitune.backend.native.models.detection.deimv2.load_checkpoint")
    def test_optimizer_configuration_structure(self, mock_load_checkpoint: MagicMock) -> None:
        """Test optimizer configuration has proper structure."""
        mock_load_checkpoint.return_value = None

        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()
        opt_config = created_model.optimizer_configuration

        # Should have 3 configurations
        assert len(opt_config) == 3

        # First config: dinov3 params excluding norm/bn/bias
        assert "params" in opt_config[0]
        assert "lr" in opt_config[0]
        assert "dinov3" in opt_config[0]["params"]

        # Second config: dinov3 norm/bn/bias with weight_decay=0
        assert "params" in opt_config[1]
        assert "lr" in opt_config[1]
        assert opt_config[1].get("weight_decay") == 0.0

        # Third config: sta/encoder/decoder norm/bn/bias with weight_decay=0
        assert "params" in opt_config[2]
        assert opt_config[2].get("weight_decay") == 0.0
