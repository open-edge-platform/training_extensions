# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Vision Transformer backbone implementation.

This module provides the DINOv3 Vision Transformer architecture with support for
Rotary Position Embeddings (RoPE), storage tokens, and various normalization options.

Modified from DEIMv2 (https://github.com/Intellindust-AI-Lab/DEIMv2)
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch import Tensor, nn

from otx.backend.native.models.classification.utils.swiglu_ffn import SwiGLUFFNV2
from otx.backend.native.models.common.layers.position_embed import RopePositionEmbedding
from otx.backend.native.models.common.layers.transformer_layers import (
    MLP2L as MLP,
)
from otx.backend.native.models.common.layers.transformer_layers import (
    LayerScale,
    SelfAttentionBlock,
)
from otx.backend.native.models.detection.layers.transformer_layers import RMSNorm
from otx.backend.native.models.modules.transformer import UnflattenPatchEmbed as PatchEmbed

if TYPE_CHECKING:
    from torch.types import Device

logger = logging.getLogger(__name__)


def named_apply(
    fn: Callable[[nn.Module, str], None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply a function recursively to all submodules of a module.

    Args:
        fn: Function to apply, takes (module, name) as arguments.
        module: Root module to start from.
        name: Name prefix for the current module. Defaults to "".
        depth_first: If True, apply to children before parent. Defaults to True.
        include_root: If True, also apply to the root module. Defaults to False.

    Returns:
        The input module (for chaining).
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class Weights(Enum):
    """Pretrained weight options for DINOv3 models."""

    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


# Model configurations for different DINOv3 variants
DINOV3_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov3_vits16": {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "pos_embed_rope_base": 100,
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_rescale_coords": 2,
        "pos_embed_rope_dtype": "fp32",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.0,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "pretrained": True,
        "weights": Weights.LVD1689M,
        "compact_arch_name": "vits",
        "check_hash": False,
    },
    "dinov3_vits16plus": {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "pos_embed_rope_base": 100,
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_rescale_coords": 2,
        "pos_embed_rope_dtype": "fp32",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 6,
        "qkv_bias": True,
        "drop_path_rate": 0.0,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "swiglu",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "pretrained": True,
        "weights": Weights.LVD1689M,
        "compact_arch_name": "vitsplus",
        "check_hash": False,
    },
}

# FFN layer type mapping
FFN_LAYER_DICT: dict[str, type[nn.Module] | partial[nn.Module]] = {
    "mlp": MLP,
    "swiglu": SwiGLUFFNV2,
    "swiglu32": partial(SwiGLUFFNV2, align_to=32),
    "swiglu64": partial(SwiGLUFFNV2, align_to=64),
    "swiglu128": partial(SwiGLUFFNV2, align_to=128),
}

# Normalization layer type mapping
NORM_LAYER_DICT: dict[str, type[nn.Module] | partial[nn.Module]] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

# Data type mapping
DTYPE_DICT: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = "") -> None:
    """Initialize weights for Vision Transformer modules.

    Args:
        module: Module to initialize.
        name: Name of the module (unused, for compatibility with named_apply).
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    """DINOv3 Vision Transformer backbone.

    A Vision Transformer with support for Rotary Position Embeddings (RoPE),
    storage tokens, and flexible normalization options. Designed for
    self-supervised learning and downstream tasks.

    Args:
        name: Model configuration name. Must be one of the keys in DINOV3_CONFIGS
            (e.g., 'dinov3_vits16', 'dinov3_vits16plus').

    Attributes:
        embed_dim: Embedding dimension.
        n_blocks: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Size of image patches.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        config = DINOV3_CONFIGS[name]

        # Extract configuration parameters
        img_size: int = config["img_size"]
        patch_size: int = config["patch_size"]
        in_chans: int = config["in_chans"]
        embed_dim: int = config["embed_dim"]
        depth: int = config["depth"]
        num_heads: int = config["num_heads"]
        ffn_ratio: int = config["ffn_ratio"]
        qkv_bias: bool = config["qkv_bias"]
        drop_path_rate: float = config["drop_path_rate"]
        layerscale_init: float = config["layerscale_init"]
        norm_layer_name: str = config["norm_layer"]
        ffn_layer_name: str = config["ffn_layer"]
        ffn_bias: bool = config["ffn_bias"]
        proj_bias: bool = config["proj_bias"]
        n_storage_tokens: int = config["n_storage_tokens"]
        mask_k_bias: bool = config["mask_k_bias"]

        # RoPE configuration
        pos_embed_rope_base: float = config["pos_embed_rope_base"]
        pos_embed_rope_normalize_coords: str = config["pos_embed_rope_normalize_coords"]
        pos_embed_rope_rescale_coords: float | None = config["pos_embed_rope_rescale_coords"]
        pos_embed_rope_dtype: str = config["pos_embed_rope_dtype"]
        pos_embed_rope_min_period: float | None = None
        pos_embed_rope_max_period: float | None = None
        pos_embed_rope_shift_coords: float | None = None
        pos_embed_rope_jitter_coords: float | None = None

        # Fixed configuration
        untie_cls_and_patch_norms: bool = False
        untie_global_and_local_cls_norm: bool = False
        device: Device | None = None

        norm_layer_cls = NORM_LAYER_DICT[norm_layer_name]

        # Store key attributes
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        # RoPE position embedding
        logger.info("RoPE config: base=%s, normalize_coords=%s, rescale_coords=%s, dtype=%s",
                    pos_embed_rope_base, pos_embed_rope_normalize_coords,
                    pos_embed_rope_rescale_coords, pos_embed_rope_dtype)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=DTYPE_DICT[pos_embed_rope_dtype],
            device=device,
        )

        # Transformer blocks
        logger.info("Using %s layer as FFN", ffn_layer_name)
        ffn_layer_cls = FFN_LAYER_DICT[ffn_layer_name]
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for _ in range(depth)
        ])
        self.chunked_blocks = False

        # Normalization layers
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm: nn.Module | None = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm: nn.Module | None = (
            norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        )

        # Head and mask token
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        self.rope_embed._init_weights()  # noqa: SLF001
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self,
        x: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, tuple[int, int]]:
        """Prepare input tokens with optional masking.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            masks: Optional boolean mask tensor for masked token prediction.

        Returns:
            Tuple of:
                - Token tensor of shape (B, 1 + n_storage_tokens + num_patches, embed_dim)
                - Tuple of (H, W) patch grid dimensions
        """
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(
        self,
        x_list: list[Tensor],
        masks_list: list[Tensor | None],
    ) -> list[dict[str, Tensor]]:
        """Process a list of inputs through the transformer.

        Args:
            x_list: List of input image tensors.
            masks_list: List of optional mask tensors.

        Returns:
            List of output dictionaries containing normalized features.
        """
        x: list[Tensor] = []
        rope: list[tuple[int, int]] = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        for blk in self.blocks:
            rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            x = blk(x, rope_sincos)

        output: list[dict[str, Tensor]] = []
        for idx, (features, masks) in enumerate(zip(x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry corresponds to local crops (training only)
                    x_norm_cls_reg = self.local_cls_norm(features[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(features[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(features[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(features[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(features)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

            output.append({
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": features,
                "masks": masks,
            })
        return output

    def forward_features(
        self,
        x: Tensor | list[Tensor],
        masks: Tensor | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Extract features from input images.

        Args:
            x: Input image tensor or list of tensors.
            masks: Optional mask tensor.

        Returns:
            Feature dictionary or list of feature dictionaries.
        """
        if isinstance(x, Tensor):
            return self.forward_features_list([x], [masks])[0]
        return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self,
        x: Tensor,
        n: int | Sequence[int] = 1,
    ) -> list[Tensor]:
        """Get intermediate layer outputs (non-chunked version).

        Args:
            x: Input image tensor.
            n: Number of last layers to return, or sequence of layer indices.

        Returns:
            List of intermediate layer output tensors.
        """
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output: list[Tensor] = []
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)

        if len(output) != len(blocks_to_take):
            raise ValueError(f"Only {len(output)} / {len(blocks_to_take)} blocks found")
        return output

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, Tensor], ...] | tuple[tuple[Tensor, Tensor, Tensor], ...]:
        """Get intermediate layer outputs with optional reshaping and token returns.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            n: Number of last layers to return, or sequence of layer indices.
            reshape: If True, reshape outputs to (B, C, H/patch, W/patch) format.
            return_class_token: If True, include class tokens in output.
            return_extra_tokens: If True, include storage tokens in output.
            norm: If True, apply normalization to outputs.

        Returns:
            Tuple of outputs. Format depends on return_class_token and return_extra_tokens:
                - Neither: tuple of patch token tensors
                - return_class_token only: tuple of (patches, cls_token)
                - return_extra_tokens only: tuple of (patches, extra_tokens)
                - Both: tuple of (patches, cls_token, extra_tokens)
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs_normed: list[Tensor] = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self,
        *args: Any,
        is_training: bool = False,
        **kwargs: Any,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]] | Tensor:
        """Forward pass through the model.

        Args:
            *args: Positional arguments passed to forward_features.
            is_training: If True, return full feature dict; otherwise return class token.
            **kwargs: Keyword arguments passed to forward_features.

        Returns:
            If is_training: Full feature dictionary or list of dictionaries.
            Otherwise: Class token output from the head.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])
