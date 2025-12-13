# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 with Spatial Token Attention (STA) backbone for DEIMv2 model.

This module provides multi-scale feature extraction by combining DINOv3/ViT-Tiny
semantic features with spatial prior features from a lightweight CNN module.

Modified from DEIMv2 (https://github.com/Intellindust-AI-Lab/DEIMv2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn.functional as f
from torch import Tensor, nn

from otx.backend.native.models.common.backbones.dinov3 import DinoVisionTransformer
from otx.backend.native.models.detection.backbones.vit_tiny import VisionTransformer

logger = logging.getLogger(__name__)


def get_norm_layer(num_features: int, use_sync_bn: bool = True) -> nn.Module:
    """Get appropriate normalization layer based on distributed training context.

    Uses SyncBatchNorm for multi-GPU training, regular BatchNorm otherwise.

    Args:
        num_features: Number of features for the normalization layer.
        use_sync_bn: If True, use SyncBatchNorm when in multi-GPU setting.

    Returns:
        BatchNorm2d or SyncBatchNorm based on training context.
    """
    if use_sync_bn and dist.is_initialized() and dist.get_world_size() > 1:
        return nn.SyncBatchNorm(num_features)
    return nn.BatchNorm2d(num_features)


class SpatialPriorModulev2(nn.Module):
    """Lightweight Spatial Prior Module for extracting multi-scale detail features.

    This module extracts fine-grained spatial details at multiple scales (1/8, 1/16, 1/32)
    using a series of convolutional layers. These features are fused with semantic
    features from the ViT backbone for improved detection performance.

    Args:
        inplanes: Base number of channels for the convolutional layers. Defaults to 16.
        use_sync_bn: Whether to use SyncBatchNorm for multi-GPU training. Defaults to True.
    """

    def __init__(self, inplanes: int = 16, use_sync_bn: bool = True) -> None:
        super().__init__()

        # 1/4 scale stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_layer(inplanes, use_sync_bn),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 1/8 scale
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_layer(2 * inplanes, use_sync_bn),
        )
        # 1/16 scale
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_layer(4 * inplanes, use_sync_bn),
        )
        # 1/32 scale
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_layer(4 * inplanes, use_sync_bn),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extract multi-scale spatial features.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Tuple of three feature tensors at scales 1/8, 1/16, and 1/32:
                - c2: Shape (B, 2*inplanes, H/8, W/8)
                - c3: Shape (B, 4*inplanes, H/16, W/16)
                - c4: Shape (B, 4*inplanes, H/32, W/32)
        """
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32

        return c2, c3, c4


class DINOv3STAsModule(nn.Module):
    """DINOv3/ViT backbone with Spatial Token Attention for multi-scale feature extraction.

    Combines semantic features from DINOv3 or ViT-Tiny backbone with spatial prior
    features from a lightweight CNN module. Produces multi-scale features suitable
    for object detection.

    Args:
        name: Model name. Use 'dinov3_*' for DINOv3 variants or other names for ViT-Tiny.
        weights_path: Path to pretrained weights. Defaults to None.
        interaction_indexes: Layer indices to extract intermediate features from.
            Defaults to empty list.
        finetune: Whether to finetune the backbone. If False, backbone is frozen.
            Defaults to True.
        embed_dim: Embedding dimension for ViT-Tiny. Defaults to 192.
        num_heads: Number of attention heads for ViT-Tiny. Defaults to 3.
        patch_size: Patch size for the ViT backbone. Defaults to 16.
        use_sta: Whether to use the Spatial Token Attention module. Defaults to True.
        conv_inplane: Base channel number for STA module. Defaults to 16.
        hidden_dim: Hidden dimension for output projection. Defaults to embed_dim.
    """

    def __init__(
        self,
        name: str,
        weights_path: str | None = None,
        interaction_indexes: list[int] | None = None,
        finetune: bool = True,
        embed_dim: int = 192,
        num_heads: int = 3,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if interaction_indexes is None:
            interaction_indexes = []

        self.dinov3: DinoVisionTransformer | VisionTransformer
        if "dinov3" in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and Path(weights_path).exists():
                logger.info("Loading checkpoint from %s...", weights_path)
                self.dinov3.load_state_dict(torch.load(weights_path))
            else:
                logger.info("Training DINOv3 from scratch...")
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=interaction_indexes,
            )
            if weights_path is not None and Path(weights_path).exists():
                logger.info("Loading checkpoint from %s...", weights_path)
                self.dinov3._model.load_state_dict(torch.load(weights_path))  # noqa: SLF001
            else:
                logger.info("Training ViT-Tiny from scratch...")

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # Initialize the spatial prior module for detail features
        self.use_sta = use_sta
        self.sta: SpatialPriorModulev2 | None = None
        if use_sta:
            logger.info("Using Lite Spatial Prior Module with inplanes=%d", conv_inplane)
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # Linear projection layers for fusing semantic and spatial features
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ]
        )
        # Normalization layers - use BatchNorm or SyncBatchNorm based on distributed context
        use_sync = dist.is_initialized() and dist.get_world_size() > 1
        self.norms = nn.ModuleList(
            [
                get_norm_layer(hidden_dim, use_sync),
                get_norm_layer(hidden_dim, use_sync),
                get_norm_layer(hidden_dim, use_sync),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extract multi-scale features from input image.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of three feature tensors at different scales:
                - c2: Features at 1/8 scale, shape (B, hidden_dim, H/8, W/8)
                - c3: Features at 1/16 scale, shape (B, hidden_dim, H/16, W/16)
                - c4: Features at 1/32 scale, shape (B, hidden_dim, H/32, W/32)
        """
        h_c, w_c = x.shape[2] // 16, x.shape[3] // 16
        bs = x.shape[0]

        # Extract semantic features from backbone
        all_layers: list[tuple[Tensor, Tensor]]
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            result = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
            all_layers = [(out, cls) for out, cls in result]  # type: ignore[misc]
        else:
            all_layers = list(self.dinov3(x))

        # Repeat single layer for all three scales if needed
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        # Process semantic features at multiple scales
        sem_feats: list[Tensor] = []
        num_scales = len(all_layers) - 2
        for i, layer_output in enumerate(all_layers):
            feat, _ = layer_output
            sem_feat = feat.transpose(1, 2).view(bs, -1, h_c, w_c).contiguous()  # [B, D, H, W]
            resize_h, resize_w = int(h_c * 2 ** (num_scales - i)), int(w_c * 2 ** (num_scales - i))
            sem_feat = f.interpolate(sem_feat, size=[resize_h, resize_w], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # Fuse semantic and spatial features
        fused_feats: list[Tensor]
        if self.use_sta and self.sta is not None:
            detail_feats = self.sta(x)
            fused_feats = [
                torch.cat([sem_feat, detail_feat], dim=1) for sem_feat, detail_feat in zip(sem_feats, detail_feats)
            ]
        else:
            fused_feats = sem_feats

        # Apply projection and normalization
        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4


class DINOv3STAs(nn.Module):
    """Factory class for creating DINOv3/ViT with Spatial Token Attention backbones.

    This class provides predefined configurations for different DEIMv2 model variants.
    Use the model_name to select a configuration:
        - 'deimv2_x': DINOv3 ViT-S/16+ (largest)
        - 'deimv2_l': DINOv3 ViT-S/16 (large)
        - 'deimv2_m': ViT-Tiny+ (medium)
        - 'deimv2_s': ViT-Tiny (small)

    Example:
        >>> backbone = DINOv3STAs("deimv2_s")
        >>> features = backbone(images)  # Returns (c2, c3, c4) multi-scale features
    """

    backbone_cfg: ClassVar[dict[str, dict[str, Any]]] = {
        "deimv2_x": {
            "name": "dinov3_vits16plus",
            "weights_path": None,
            "interaction_indexes": [5, 8, 11],
            "conv_inplane": 64,
            "hidden_dim": 256,
        },
        "deimv2_l": {
            "name": "dinov3_vits16",
            "weights_path": None,
            "interaction_indexes": [5, 8, 11],
            "conv_inplane": 32,
            "hidden_dim": 224,
        },
        "deimv2_m": {
            "name": "vit_tinyplus",
            "embed_dim": 256,
            "weights_path": None,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 4,
        },
        "deimv2_s": {
            "name": "vit_tiny",
            "embed_dim": 192,
            "weights_path": None,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 3,
        },
    }

    def __new__(cls, model_name: str) -> DINOv3STAsModule:
        """Create a DINOv3STAs backbone instance.

        Args:
            model_name: Name of the model configuration to use.
                Must be one of: 'deimv2_x', 'deimv2_l', 'deimv2_m', 'deimv2_s'.

        Returns:
            Configured DINOv3STAsModule backbone instance.

        Raises:
            KeyError: If model_name is not in backbone_cfg.
        """
        cfg = cls.backbone_cfg[model_name].copy()
        return DINOv3STAsModule(**cfg)
