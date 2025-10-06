# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIM DFine detection model."""

from __future__ import annotations

import pytest
import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.detection.deim import DEIMDFine
from otx.data.entity.torch import OTXPredBatch


class TestDEIMDFine:
    """Test class for DEIM DFine detection model."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "deim_dfine_hgnetv2_n",
            "deim_dfine_hgnetv2_s",
            "deim_dfine_hgnetv2_m",
            "deim_dfine_hgnetv2_l",
            "deim_dfine_hgnetv2_x",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test DEIM DFine model initialization."""
        model = DEIMDFine(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert model.model_name == model_name
        assert model.num_classes == 3
        assert model.data_input_params.input_size == (640, 640)
        assert model.input_size_multiplier == 32
        assert model_name in model.pretrained_weights

    def test_create_model(self) -> None:
        """Test DEIM DFine model creation."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=10,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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

    def test_backbone_lr_mapping(self) -> None:
        """Test that backbone learning rate mapping works correctly."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_n",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        created_model = model._create_model()

        # Check optimizer configuration exists
        assert hasattr(created_model, "optimizer_configuration")
        assert len(created_model.optimizer_configuration) > 0

        # For 'n' variant, should have 3 configurations
        if model.model_name == "deim_dfine_hgnetv2_n":
            assert len(created_model.optimizer_configuration) == 3
        else:
            assert len(created_model.optimizer_configuration) == 2

    @pytest.mark.parametrize(
        ("model_name", "label_info"),
        [
            ("deim_dfine_hgnetv2_n", 3),
            ("deim_dfine_hgnetv2_s", 5),
            ("deim_dfine_hgnetv2_m", 10),
            ("deim_dfine_hgnetv2_l", 80),
            ("deim_dfine_hgnetv2_x", 20),
        ],
    )
    def test_loss_computation(self, model_name: str, label_info: int, fxt_data_module) -> None:
        """Test DEIM DFine loss computation in training mode."""
        model = DEIMDFine(
            model_name=model_name,
            label_info=label_info,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Get data batch
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, 640, 640)

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(data)

        # Check that output contains expected DEIM loss components
        assert isinstance(output, dict)
        expected_losses = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_mal"]

        for loss_name in expected_losses:
            assert loss_name in output
            assert isinstance(output[loss_name], torch.Tensor)

    @pytest.mark.parametrize(
        "model_name",
        [
            "deim_dfine_hgnetv2_n",
            "deim_dfine_hgnetv2_s",
            "deim_dfine_hgnetv2_m",
            "deim_dfine_hgnetv2_l",
            "deim_dfine_hgnetv2_x",
        ],
    )
    def test_predict(self, model_name: str, fxt_data_module) -> None:
        """Test DEIM DFine prediction in evaluation mode."""
        model = DEIMDFine(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Get data batch
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, 640, 640)

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
            "deim_dfine_hgnetv2_s",
            "deim_dfine_hgnetv2_m",
        ],
    )
    def test_export(self, model_name: str) -> None:
        """Test DEIM DFine export functionality."""
        model = DEIMDFine(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
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

    def test_multi_scale_training(self) -> None:
        """Test DEIM DFine with multi-scale training enabled."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            multi_scale=True,
        )

        # Multi-scale should be created in the model
        created_model = model._create_model()
        assert isinstance(created_model.multi_scale, list)
        assert len(created_model.multi_scale) > 0

    def test_torch_compile_integration(self) -> None:
        """Test DEIM DFine with torch compile enabled."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=True,
        )

        # Check that torch compile is enabled
        assert model.torch_compile is True

    def test_weight_dict_configuration(self) -> None:
        """Test that the weight dictionary is properly configured."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()
        criterion = created_model.criterion

        # Check that weight dict contains expected keys
        expected_weights = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf", "loss_mal"]
        for weight_key in expected_weights:
            assert weight_key in criterion.weight_dict

        # Check specific weight values
        assert criterion.weight_dict["loss_vfl"] == 1
        assert criterion.weight_dict["loss_bbox"] == 5
        assert criterion.weight_dict["loss_giou"] == 2
        assert criterion.weight_dict["loss_fgl"] == 0.15
        assert criterion.weight_dict["loss_ddf"] == 1.5
        assert criterion.weight_dict["loss_mal"] == 1.0

    def test_criterion_parameters(self) -> None:
        """Test that the criterion is configured with correct parameters."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=10,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        created_model = model._create_model()
        criterion = created_model.criterion

        # Check criterion parameters
        assert criterion.alpha == 0.75
        assert criterion.gamma == 1.5
        assert criterion.reg_max == 32
        assert criterion.num_classes == 10

    def test_dummy_input_generation(self) -> None:
        """Test dummy input generation for different batch sizes."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            dummy_input = model.get_dummy_input(batch_size)
            assert len(dummy_input.images) == batch_size
            assert dummy_input.images[0].shape == (3, 640, 640)

    def test_model_properties(self) -> None:
        """Test various model properties."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_m",
            label_info=20,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Test input size multiplier
        assert model.input_size_multiplier == 32

        # Test pretrained weights availability
        assert model.model_name in model.pretrained_weights
        assert isinstance(model.pretrained_weights[model.model_name], str)
        assert model.pretrained_weights[model.model_name].startswith("https://")

    def test_inheritance_from_rtdetr(self) -> None:
        """Test that DEIM DFine properly inherits from RTDETR."""
        from otx.backend.native.models.detection.rtdetr import RTDETR

        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Check inheritance
        assert isinstance(model, RTDETR)

        # Check that it has inherited methods
        assert hasattr(model, "forward")
        assert hasattr(model, "training_step")
        assert hasattr(model, "validation_step")
        assert hasattr(model, "predict_step")
