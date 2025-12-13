# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""DINOv3 Vision Transformer backbone implementation.

This module implements the DINOv3 Vision Transformer architecture with
RoPE (Rotary Position Embedding) and various configuration options.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from typing import Any, Callable

import torch
import torch.nn.init
from torch import Tensor, nn

from otx.backend.native.models.classification.utils.swiglu_ffn import SwiGLUFFNV2
from otx.backend.native.models.common.layers.position_embed import RopePositionEmbedding
from otx.backend.native.models.common.layers.transformer_layers import MLP2L, LayerScale, SelfAttentionBlock
from otx.backend.native.models.modules.transformer import UnflattenPatchEmbed as PatchEmbed


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply a function to all named modules recursively.

    Args:
        fn: Function to apply, should accept `module` and `name` kwargs.
        module: Root module to start from.
        name: Name prefix for the root module.
        depth_first: If True, apply in depth-first order.
        include_root: If True, also apply to the root module.

    Returns:
        The input module (for chaining).
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=full_name,
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


#: Configuration dictionary mapping model names to their hyperparameters.
configs: dict[str, dict[str, Any]] = {
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

logger = logging.getLogger("dinov3")

#: Mapping from string FFN layer names to their class implementations.
ffn_layer_dict: dict[str, type | partial] = {
    "mlp": MLP2L,
    "swiglu": SwiGLUFFNV2,
    "swiglu32": partial(SwiGLUFFNV2, align_to=32),
    "swiglu64": partial(SwiGLUFFNV2, align_to=64),
    "swiglu128": partial(SwiGLUFFNV2, align_to=128),
}

#: Mapping from string norm layer names to their class implementations.
norm_layer_dict: dict[str, type | partial] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

#: Mapping from string dtype names to torch dtype objects.
dtype_dict: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = "") -> None:  # noqa: ARG001
    """Initialize Vision Transformer module weights.

    Applies truncated normal initialization to Linear layers, and calls
    reset_parameters on LayerNorm, LayerScale and PatchEmbed.

    Args:
        module: The module to initialize.
        name: Name of the module (unused, for compatibility with named_apply).
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    """DINOv3 Vision Transformer backbone.

    A Vision Transformer with RoPE (Rotary Position Embedding), optional
    SwiGLU FFN layers, and LayerScale. Designed for self-supervised learning
    with the DINOv3 methodology.

    Args:
        name: Model configuration name from the configs dictionary.
            Supported: 'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16',
            'dinov3_vitb16plus', 'dinov3_vitl16plus'.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        config = configs[name]
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        in_chans = config["in_chans"]
        pos_embed_rope_min_period = None
        pos_embed_rope_max_period = None
        pos_embed_rope_shift_coords = None
        pos_embed_rope_jitter_coords = None
        pos_embed_rope_rescale_coords = None
        pos_embed_rope_base = config["pos_embed_rope_base"]
        pos_embed_rope_normalize_coords = config["pos_embed_rope_normalize_coords"]
        pos_embed_rope_rescale_coords = config["pos_embed_rope_rescale_coords"]
        pos_embed_rope_dtype = config["pos_embed_rope_dtype"]
        embed_dim = config["embed_dim"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        ffn_ratio = config["ffn_ratio"]
        qkv_bias = config["qkv_bias"]
        drop_path_rate = config["drop_path_rate"]
        layerscale_init = config["layerscale_init"]
        norm_layer = config["norm_layer"]
        ffn_layer = config["ffn_layer"]
        ffn_bias = config["ffn_bias"]
        proj_bias = config["proj_bias"]
        n_storage_tokens = config["n_storage_tokens"]
        mask_k_bias = config["mask_k_bias"]
        untie_cls_and_patch_norms = False
        untie_global_and_local_cls_norm = False
        device = None

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

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
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
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
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights with proper initialization schemes."""
        self.rope_embed._init_weights()  # noqa: SLF001
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Tensor | None = None) -> tuple[Tensor, tuple[int, int]]:
        """Prepare input tokens with optional mask tokens.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            masks: Optional boolean mask tensor for masked image modeling.

        Returns:
            Tuple of (tokens, (H, W)) where tokens has shape (B, N, D) with
            cls_token, storage_tokens, and patch tokens concatenated.
        """
        x = self.patch_embed(x)
        B, H, W, _ = x.shape  # noqa: N806
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
                1,
                0,
                cls_token.shape[-1],
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

    def forward_features_list(self, x_list: list[Tensor], masks_list: list[Tensor | None]) -> list[dict[str, Tensor]]:
        """Forward pass for a list of images with masks.

        Args:
            x_list: List of input image tensors.
            masks_list: List of corresponding mask tensors (can be None).

        Returns:
            List of dictionaries containing normalized features.
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(h=h, w=w) for h, w in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])  # type: ignore[call-overload]
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])  # type: ignore[call-overload]
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])  # type: ignore[call-overload]
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])  # type: ignore[call-overload]
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]  # type: ignore[call-overload]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]  # type: ignore[call-overload]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: Tensor | list[Tensor],
        masks: Tensor | list[Tensor] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Extract features from input images.

        Args:
            x: Input image tensor or list of tensors.
            masks: Optional mask tensor for masked image modeling.

        Returns:
            Dictionary (single tensor) or list of dictionaries containing
            normalized CLS token, storage tokens, patch tokens, and pre-norm features.
        """
        if isinstance(x, torch.Tensor):
            masks_as_list: list[Tensor | None] = [masks]
            return self.forward_features_list([x], masks_as_list)[0]
        return self.forward_features_list(x, masks if masks is not None else [None] * len(x))

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int | list[int] = 1) -> list[Tensor]:
        """Get intermediate layer outputs without chunking.

        Args:
            x: Input tensor.
            n: Number of last layers to return, or list of layer indices.

        Returns:
            List of intermediate feature tensors.
        """
        x, (h, w) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output: list[Tensor] = []
        total_block_len = len(self.blocks)
        blocks_to_take: range | list[int] = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(h=h, w=w) if self.rope_embed is not None else None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        if len(output) != len(blocks_to_take):
            msg = f"only {len(output)} / {len(blocks_to_take)} blocks found"
            raise RuntimeError(msg)
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: int | list[int] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]:
        """Get intermediate layer representations.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            n: Number of last layers to return, or list of specific layer indices.
            reshape: If True, reshape outputs to spatial format (B, C, H', W').
            return_class_token: If True, also return class tokens.
            return_extra_tokens: If True, also return extra/storage tokens.
            norm: If True, apply layer normalization to outputs.

        Returns:
            Tuple of outputs. Format depends on return flags:
            - Default: (outputs,) for each layer
            - With class token: ((output, cls_token),) for each layer
            - With extra tokens: ((output, extra_tokens),) for each layer
            - Both: ((output, cls_token, extra_tokens),) for each layer
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
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
            b, _, h, w = x.shape
            outputs = [
                out.reshape(b, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        if return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))
        return tuple(outputs)

    def forward(
        self,
        *args: Any,  # noqa: ANN401
        is_training: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Tensor] | list[dict[str, Tensor]] | Tensor:
        """Forward pass through the model.

        Args:
            *args: Positional arguments passed to forward_features.
            is_training: If True, return full feature dict; otherwise return
                classification head output.
            **kwargs: Keyword arguments passed to forward_features.

        Returns:
            Feature dictionary during training, or head output during inference.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        if isinstance(ret, list):
            ret = ret[0]
        return self.head(ret["x_norm_clstoken"])
