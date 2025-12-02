# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DINOv3 Vision Transformer backbone."""

import pytest
import torch

from otx.backend.native.models.common.backbones.dinov3 import (
    DinoVisionTransformer,
    Weights,
    configs,
    dtype_dict,
    ffn_layer_dict,
    init_weights_vit,
    named_apply,
    norm_layer_dict,
)


class TestDinoVisionTransformer:
    """Test class for DinoVisionTransformer."""

    @pytest.fixture
    def vit_small(self):
        """Create a ViT-S/16 model."""
        return DinoVisionTransformer(name="dinov3_vits16")

    @pytest.fixture
    def vit_small_plus(self):
        """Create a ViT-S/16+ model with SwiGLU."""
        return DinoVisionTransformer(name="dinov3_vits16plus")

    def test_init_vits16(self, vit_small):
        """Test ViT-S/16 initialization."""
        assert isinstance(vit_small, DinoVisionTransformer)
        assert vit_small.embed_dim == 384
        assert vit_small.num_features == 384
        assert vit_small.n_blocks == 12
        assert vit_small.num_heads == 6
        assert vit_small.patch_size == 16
        assert vit_small.n_storage_tokens == 4

    def test_init_vits16plus(self, vit_small_plus):
        """Test ViT-S/16+ initialization with SwiGLU."""
        assert isinstance(vit_small_plus, DinoVisionTransformer)
        assert vit_small_plus.embed_dim == 384
        assert vit_small_plus.n_blocks == 12
        # Check SwiGLU FFN is used
        from otx.backend.native.models.classification.utils.swiglu_ffn import SwiGLUFFNV2

        assert isinstance(vit_small_plus.blocks[0].mlp, SwiGLUFFNV2)

    def test_model_components(self, vit_small):
        """Test that all model components are properly initialized."""
        # Check patch embedding
        assert hasattr(vit_small, "patch_embed")
        assert vit_small.patch_embed.patch_size == (16, 16)

        # Check cls token
        assert hasattr(vit_small, "cls_token")
        assert vit_small.cls_token.shape == (1, 1, 384)

        # Check storage tokens
        assert hasattr(vit_small, "storage_tokens")
        assert vit_small.storage_tokens.shape == (1, 4, 384)

        # Check RoPE embedding
        assert hasattr(vit_small, "rope_embed")

        # Check transformer blocks
        assert hasattr(vit_small, "blocks")
        assert len(vit_small.blocks) == 12

        # Check normalization
        assert hasattr(vit_small, "norm")

        # Check mask token
        assert hasattr(vit_small, "mask_token")
        assert vit_small.mask_token.shape == (1, 384)

    def test_forward_single_image(self, vit_small):
        """Test forward pass with a single image."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = vit_small(x)

        # In eval mode, returns cls token features
        assert output.shape == (1, 384)

    def test_forward_batch(self, vit_small):
        """Test forward pass with a batch of images."""
        vit_small.eval()
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = vit_small(x)

        assert output.shape == (batch_size, 384)

    def test_forward_features(self, vit_small):
        """Test forward_features method."""
        vit_small.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = vit_small.forward_features(x)

        assert isinstance(output, dict)
        assert "x_norm_clstoken" in output
        assert "x_storage_tokens" in output
        assert "x_norm_patchtokens" in output
        assert "x_prenorm" in output
        assert "masks" in output

        # Check shapes
        assert output["x_norm_clstoken"].shape == (2, 384)
        assert output["x_storage_tokens"].shape == (2, 4, 384)
        # 224/16 = 14, so 14*14 = 196 patch tokens
        assert output["x_norm_patchtokens"].shape == (2, 196, 384)

    def test_forward_features_list(self, vit_small):
        """Test forward_features with list of images."""
        vit_small.eval()
        x1 = torch.randn(2, 3, 224, 224)
        x2 = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = vit_small.forward_features([x1, x2])

        assert isinstance(output, list)
        assert len(output) == 2
        for out in output:
            assert isinstance(out, dict)
            assert "x_norm_clstoken" in out

    def test_forward_training_mode(self, vit_small):
        """Test forward pass in training mode."""
        vit_small.train()
        x = torch.randn(2, 3, 224, 224)

        output = vit_small(x, is_training=True)

        assert isinstance(output, dict)
        assert "x_norm_clstoken" in output

    def test_prepare_tokens_with_masks(self, vit_small):
        """Test token preparation with masks."""
        x = torch.randn(2, 3, 224, 224)

        tokens, (h, w) = vit_small.prepare_tokens_with_masks(x)

        # Should have cls_token + storage_tokens + patch_tokens
        # 1 + 4 + 196 = 201
        assert tokens.shape == (2, 201, 384)
        assert h == 14
        assert w == 14

    def test_prepare_tokens_with_mask_tokens(self, vit_small):
        """Test token preparation with mask tokens applied."""
        x = torch.randn(2, 3, 224, 224)
        # Create a mask for some patch positions
        masks = torch.zeros(2, 196, dtype=torch.bool)
        masks[:, :50] = True  # Mask first 50 patches

        tokens, (h, w) = vit_small.prepare_tokens_with_masks(x, masks)

        assert tokens.shape == (2, 201, 384)
        assert h == 14
        assert w == 14

    def test_get_intermediate_layers(self, vit_small):
        """Test getting intermediate layer outputs."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            # Get last 2 layers
            outputs = vit_small.get_intermediate_layers(x, n=2)

        assert len(outputs) == 2
        for out in outputs:
            assert out.shape == (1, 196, 384)

    def test_get_intermediate_layers_specific_indices(self, vit_small):
        """Test getting specific intermediate layers by index."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            # Get layers 5 and 10
            outputs = vit_small.get_intermediate_layers(x, n=[5, 10])

        assert len(outputs) == 2

    def test_get_intermediate_layers_reshape(self, vit_small):
        """Test getting intermediate layers with spatial reshape."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = vit_small.get_intermediate_layers(x, n=1, reshape=True)

        assert len(outputs) == 1
        # Should be reshaped to (B, C, H, W)
        assert outputs[0].shape == (1, 384, 14, 14)

    def test_get_intermediate_layers_with_cls_token(self, vit_small):
        """Test getting intermediate layers with class token."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = vit_small.get_intermediate_layers(x, n=1, return_class_token=True)

        assert len(outputs) == 1
        assert len(outputs[0]) == 2  # (features, cls_token)
        features, cls_token = outputs[0]
        assert features.shape == (1, 196, 384)
        assert cls_token.shape == (1, 384)

    def test_get_intermediate_layers_with_extra_tokens(self, vit_small):
        """Test getting intermediate layers with extra/storage tokens."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = vit_small.get_intermediate_layers(x, n=1, return_extra_tokens=True)

        assert len(outputs) == 1
        assert len(outputs[0]) == 2  # (features, extra_tokens)
        features, extra_tokens = outputs[0]
        assert features.shape == (1, 196, 384)
        assert extra_tokens.shape == (1, 4, 384)

    def test_get_intermediate_layers_with_both_tokens(self, vit_small):
        """Test getting intermediate layers with both cls and extra tokens."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = vit_small.get_intermediate_layers(x, n=1, return_class_token=True, return_extra_tokens=True)

        assert len(outputs) == 1
        assert len(outputs[0]) == 3  # (features, cls_token, extra_tokens)
        features, cls_token, extra_tokens = outputs[0]
        assert features.shape == (1, 196, 384)
        assert cls_token.shape == (1, 384)
        assert extra_tokens.shape == (1, 4, 384)

    def test_get_intermediate_layers_no_norm(self, vit_small):
        """Test getting intermediate layers without normalization."""
        vit_small.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs_normed = vit_small.get_intermediate_layers(x, n=1, norm=True)
            outputs_raw = vit_small.get_intermediate_layers(x, n=1, norm=False)

        # Outputs should be different
        assert not torch.allclose(outputs_normed[0], outputs_raw[0])

    def test_different_input_sizes(self, vit_small):
        """Test forward pass with different input sizes."""
        vit_small.eval()

        # Test with 448x448 (larger than training size)
        x_large = torch.randn(1, 3, 448, 448)
        with torch.no_grad():
            output_large = vit_small.forward_features(x_large)

        # 448/16 = 28, so 28*28 = 784 patches
        assert output_large["x_norm_patchtokens"].shape == (1, 784, 384)

    def test_init_weights(self, vit_small):
        """Test weight initialization."""
        # Simply verify init_weights runs without error
        vit_small.init_weights()

        # Check that cls_token has non-zero values (initialized)
        assert not torch.all(vit_small.cls_token == 0)

        # Check that mask_token is zero-initialized
        assert torch.all(vit_small.mask_token == 0)


class TestHelperFunctions:
    """Test helper functions and configurations."""

    def test_configs_exist(self):
        """Test that configurations exist."""
        assert "dinov3_vits16" in configs
        assert "dinov3_vits16plus" in configs

    def test_configs_structure(self):
        """Test configuration structure."""
        config = configs["dinov3_vits16"]

        required_keys = [
            "img_size",
            "patch_size",
            "embed_dim",
            "depth",
            "num_heads",
            "ffn_ratio",
            "norm_layer",
            "ffn_layer",
        ]
        for key in required_keys:
            assert key in config

    def test_weights_enum(self):
        """Test Weights enum."""
        assert Weights.LVD1689M.value == "LVD1689M"
        assert Weights.SAT493M.value == "SAT493M"

    def test_ffn_layer_dict(self):
        """Test FFN layer mapping."""
        assert "mlp" in ffn_layer_dict
        assert "swiglu" in ffn_layer_dict
        assert "swiglu32" in ffn_layer_dict
        assert "swiglu64" in ffn_layer_dict
        assert "swiglu128" in ffn_layer_dict

    def test_norm_layer_dict(self):
        """Test norm layer mapping."""
        assert "layernorm" in norm_layer_dict
        assert "layernormbf16" in norm_layer_dict

    def test_dtype_dict(self):
        """Test dtype mapping."""
        assert dtype_dict["fp32"] == torch.float32
        assert dtype_dict["fp16"] == torch.float16
        assert dtype_dict["bf16"] == torch.bfloat16

    def test_named_apply(self):
        """Test named_apply function."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )

        applied_names = []

        def track_fn(module, name) -> None:
            applied_names.append(name)

        named_apply(track_fn, model, include_root=True)

        # Should have applied to root and all children
        assert "" in applied_names  # Root
        assert "0" in applied_names
        assert "1" in applied_names
        assert "2" in applied_names

    def test_named_apply_depth_first(self):
        """Test named_apply with depth-first ordering."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 5),
        )

        order = []

        def track_order(module, name) -> None:
            order.append(name)

        # Depth-first: children before parent
        named_apply(track_order, model, depth_first=True, include_root=True)

        # Children should come before root
        root_idx = order.index("")
        child_idx = order.index("0")
        assert child_idx < root_idx

    def test_named_apply_breadth_first(self):
        """Test named_apply with breadth-first ordering."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 5),
        )

        order = []

        def track_order(module, name) -> None:
            order.append(name)

        # Breadth-first: parent before children
        named_apply(track_order, model, depth_first=False, include_root=True)

        # Root should come before children
        root_idx = order.index("")
        child_idx = order.index("0")
        assert root_idx < child_idx

    def test_init_weights_vit_linear(self):
        """Test init_weights_vit for Linear layers."""
        linear = torch.nn.Linear(10, 20)
        original_weight = linear.weight.clone()

        init_weights_vit(linear)

        # Weight should be reinitialized
        assert not torch.allclose(linear.weight, original_weight)
        # Bias should be zeros
        assert torch.all(linear.bias == 0)

    def test_init_weights_vit_layernorm(self):
        """Test init_weights_vit for LayerNorm."""
        ln = torch.nn.LayerNorm(256)

        # Should not raise error
        init_weights_vit(ln)

        # LayerNorm should have default initialization
        assert torch.allclose(ln.weight, torch.ones(256))
        assert torch.allclose(ln.bias, torch.zeros(256))


class TestGradients:
    """Test gradient flow through the model."""

    @pytest.fixture
    def vit_small(self):
        """Create a ViT-S/16 model for gradient testing."""
        return DinoVisionTransformer(name="dinov3_vits16")

    def test_gradient_flow(self, vit_small):
        """Test that gradients flow through the model."""
        vit_small.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)

        output = vit_small(x, is_training=True)
        loss = output["x_norm_clstoken"].sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_to_parameters(self, vit_small):
        """Test that gradients flow to all trainable parameters."""
        vit_small.train()
        x = torch.randn(2, 3, 224, 224)

        output = vit_small(x, is_training=True)
        loss = output["x_norm_clstoken"].sum()
        loss.backward()

        # Check key parameters have gradients
        assert vit_small.cls_token.grad is not None
        assert vit_small.storage_tokens.grad is not None

        # Check some block parameters
        assert vit_small.blocks[0].attn.qkv.weight.grad is not None
