# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RF-DETR instance segmentation model."""

from __future__ import annotations

import torch

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.instance_segmentation.rfdetr_inst import RFDETRInst
from getitune.data.entity import PredictionBatch


class TestRFDETRInst:
    """Test class for RF-DETR instance segmentation model."""

    def test_init(self) -> None:
        """Test RF-DETR instance segmentation model initialization."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
        )
        assert model.model_name == "rfdetr_seg_n"
        assert model.num_classes == 3

    def test_create_model(self) -> None:
        """Test RF-DETR instance segmentation model creation."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
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
        """Test default preprocessing parameters for RF-DETR instance segmentation."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
        )

        # Check that default params use 0-1 range normalization
        default_params = model._default_preprocessing_params
        assert "rfdetr_seg_n" in default_params
        assert default_params["rfdetr_seg_n"].input_size == (312, 312)
        # ImageNet mean in 0-1 range
        assert default_params["rfdetr_seg_n"].mean == (0.485, 0.456, 0.406)
        assert default_params["rfdetr_seg_n"].std == (0.229, 0.224, 0.225)

    def test_optimizer_configuration(self) -> None:
        """Test that optimizer configuration is properly set."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=5,
        )

        # Test configure_optimizers method
        optimizers, schedulers = model.configure_optimizers()

        assert len(optimizers) == 1
        assert isinstance(optimizers[0], torch.optim.Optimizer)
        assert len(schedulers) > 0
        assert isinstance(schedulers, list)

        # Check that parameter groups are properly configured
        param_groups = optimizers[0].param_groups
        assert len(param_groups) > 0

    def test_loss_computation(self, fxt_instance_seg_batch) -> None:
        """Test RF-DETR instance segmentation loss computation in training mode."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            data_input_params=DataInputParams((312, 312), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Adjust batch images to match input size
        fxt_instance_seg_batch.images = [
            torch.randn(3, 312, 312),
            torch.randn(3, 312, 312),
        ]

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(fxt_instance_seg_batch)

        # Check that output contains loss components
        assert isinstance(output, dict)
        # RF-DETR segmentation should have mask losses in addition to detection losses
        assert any("loss" in key for key in output)

        # Check that loss values are not None and are valid tensors
        for key, value in output.items():
            if "loss" in key:
                assert value is not None
                assert isinstance(value, torch.Tensor)
                assert not torch.isnan(value)
                assert not torch.isinf(value)

    def test_predict(self, fxt_instance_seg_batch) -> None:
        """Test RF-DETR instance segmentation prediction in evaluation mode."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            data_input_params=DataInputParams((312, 312), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Adjust batch images to match input size
        fxt_instance_seg_batch.images = [
            torch.randn(3, 312, 312),
            torch.randn(3, 312, 312),
        ]

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(fxt_instance_seg_batch)

        # Check that output is PredictionBatch with masks
        assert isinstance(output, PredictionBatch)
        assert output.batch_size == 2
        assert output.masks is not None
        assert len(output.masks) == 2

    def test_export(self) -> None:
        """Test RF-DETR instance segmentation export functionality."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, 312, 312))
        # Should return (boxes, labels, masks) where ``boxes`` has scores
        # concatenated as the 5th column to match the OpenVINO ``MaskRCNN``
        # model_api wrapper's expectation of ``boxes[:, 4]`` being the score.
        assert len(output) == 3
        boxes, labels, masks = output
        assert boxes.ndim == 3
        assert boxes.shape[-1] == 5  # x1, y1, x2, y2, score
        assert labels.shape[:2] == boxes.shape[:2]
        assert masks.shape[:2] == boxes.shape[:2]

    def test_exporter_output_names(self) -> None:
        """Exporter must publish ``boxes``/``labels``/``masks`` (no standalone ``scores``).

        The OpenVINO ``MaskRCNN`` model_api wrapper used at test/optimize time
        reads ``outputs["boxes"][:, 4]`` for the score; emitting a separate
        ``scores`` output and a 4-column ``boxes`` tensor would raise
        ``IndexError: index 4 is out of bounds for axis 1 with size 4``.
        """
        model = RFDETRInst(model_name="rfdetr_seg_n", label_info=3)
        exporter = model._exporter
        assert exporter.output_names == ["boxes", "labels", "masks"]
        assert isinstance(exporter, LightningModelExporter)
        onnx_cfg = exporter.onnx_export_configuration
        assert onnx_cfg["output_names"] == ["boxes", "labels", "masks"]

    def test_customize_inputs(self, fxt_instance_seg_batch) -> None:
        """Test input customization for RF-DETR format."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
        )

        customized = model._customize_inputs(fxt_instance_seg_batch)

        # Check that customized inputs have the expected format
        assert "images" in customized
        assert "targets" in customized
        assert isinstance(customized["targets"], list)
        assert len(customized["targets"]) == fxt_instance_seg_batch.batch_size

        # Check target structure
        for target in customized["targets"]:
            assert "boxes" in target
            assert "labels" in target
            assert "masks" in target
            assert "size" in target
            assert "orig_size" in target
