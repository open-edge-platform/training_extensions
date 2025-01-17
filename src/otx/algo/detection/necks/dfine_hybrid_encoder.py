# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""D-FINE Hybrid Encoder. Modified from D-FINE (https://github.com/Peterande/D-FINE)."""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, ClassVar

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.common.layers.transformer_layers import TransformerEncoder, TransformerEncoderLayer
from otx.algo.detection.layers.csp_layer import CSPRepLayer
from otx.algo.detection.utils.utils import auto_pad
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer


class SCDown(nn.Module):
    """SCDown downsampling module.

    Args:
        c1 (int): Number of channels in the input feature map.
        c2 (int): Number of channels produced by the convolution.
        k (int): Kernel size of the convolving kernel.
        s (int): Stride of the convolution.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int,
        s: int,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.cv1 = Conv2dModule(
            c1,
            c2,
            1,
            1,
            normalization=build_norm_layer(normalization, num_features=c2),
            activation=None,
        )
        self.cv2 = Conv2dModule(
            c2,
            c2,
            k,
            s,
            padding=auto_pad(kernel_size=k),
            groups=c2,
            normalization=build_norm_layer(normalization, num_features=c2),
            activation=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.cv2(self.cv1(x))


class RepNCSPELAN4(nn.Module):
    """GELANModule from YOLOv9.

    Note:
        Might not be replaceable as layer implementation is very different from GELANModule in YOLOv9.

    Args:
        c1 (int): c1 channel size. Refer to GELAN paper.
        c2 (int): c2 channel size. Refer to GELAN paper.
        c3 (int): c3 channel size. Refer to GELAN paper.
        c4 (int): c4 channel size. Refer to GELAN paper.
        n (int, optional): number of blocks. Defaults to 3.
        bias (bool, optional): use bias. Defaults to False.
        activation (Callable[..., nn.Module] | None, optional): activation function. Defaults to None.
        normalization (Callable[..., nn.Module] | None, optional): norm layer. Defaults to None.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        num_blocks: int = 3,
        bias: bool = False,
        activation: Callable[..., nn.Module] | None = None,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.c = c3 // 2

        self.cv1 = Conv2dModule(
            c1,
            c3,
            1,
            1,
            bias=bias,
            activation=build_activation_layer(activation),
            normalization=build_norm_layer(normalization, num_features=c3),
        )

        self.cv2 = nn.Sequential(
            CSPRepLayer(
                c3 // 2,
                c4,
                num_blocks,
                1,
                bias=bias,
                activation=activation,
                normalization=normalization,
            ),
            Conv2dModule(
                c4,
                c4,
                3,
                1,
                padding=auto_pad(kernel_size=3),
                bias=bias,
                activation=build_activation_layer(activation),
                normalization=build_norm_layer(normalization, num_features=c4),
            ),
        )

        self.cv3 = nn.Sequential(
            CSPRepLayer(
                c4,
                c4,
                num_blocks,
                1,
                bias=bias,
                activation=activation,
                normalization=normalization,
            ),
            Conv2dModule(
                c4,
                c4,
                3,
                1,
                padding=auto_pad(kernel_size=3),
                bias=bias,
                activation=build_activation_layer(activation),
                normalization=build_norm_layer(normalization, num_features=c4),
            ),
        )

        self.cv4 = Conv2dModule(
            c3 + (2 * c4),
            c2,
            1,
            1,
            bias=bias,
            activation=build_activation_layer(activation),
            normalization=build_norm_layer(normalization, num_features=c2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class HybridEncoderModule(nn.Module):
    """HybridEncoder for DFine.

    TODO(Eugene): Merge with current rtdetr.HybridEncoderModule in next PR.

    Args:
        in_channels (list[int], optional): List of input channels for each feature map.
            Defaults to [512, 1024, 2048].
        feat_strides (list[int], optional): List of stride values for
            each feature map. Defaults to [8, 16, 32].
        hidden_dim (int, optional): Hidden dimension size. Defaults to 256.
        nhead (int, optional): Number of attention heads in the transformer encoder.
                Defaults to 8.
        dim_feedforward (int, optional): Dimension of the feedforward network
            in the transformer encoder. Defaults to 1024.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        enc_activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm")``.
        use_encoder_idx (list[int], optional): List of indices of the encoder to use.
            Defaults to [2].
        num_encoder_layers (int, optional): Number of layers in the transformer encoder.
            Defaults to 1.
        pe_temperature (float, optional): Temperature parameter for positional encoding.
            Defaults to 10000.
        expansion (float, optional): Expansion factor for the CSPRepLayer.
            Defaults to 1.0.
        depth_mult (float, optional): Depth multiplier for the CSPRepLayer.
            Defaults to 1.0.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.SiLU``.
        eval_spatial_size (tuple[int, int] | None, optional): Spatial size for
            evaluation. Defaults to None.
    """

    def __init__(
        self,
        in_channels: list[int] = [512, 1024, 2048],  # noqa: B006
        feat_strides: list[int] = [8, 16, 32],  # noqa: B006
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm"),
        use_encoder_idx: list[int] = [2],  # noqa: B006
        num_encoder_layers: int = 1,
        pe_temperature: int = 10000,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        activation: Callable[..., nn.Module] = nn.SiLU,
        eval_spatial_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ],
                    ),
                ),
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_activation,
        )

        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))],
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                Conv2dModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    normalization=build_norm_layer(normalization, num_features=hidden_dim),
                    activation=None,
                ),
            )
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                    activation=activation,
                    normalization=normalization,
                ),
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    SCDown(
                        hidden_dim,
                        hidden_dim,
                        3,
                        2,
                        normalization=normalization,
                    ),
                ),
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                    activation=activation,
                    normalization=normalization,
                ),
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
    ) -> Tensor:
        """Build 2D sin-cos position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            msg = "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
            raise ValueError(msg)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats: Tensor) -> list[Tensor]:
        """Forward pass."""
        if len(feats) != len(self.in_channels):
            msg = f"Input feature size {len(feats)} does not match the number of input channels {len(self.in_channels)}"
            raise ValueError(msg)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(
                        src_flatten.device,
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}").to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = f.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


class HybridEncoder:
    """HybridEncoder factory for D-Fine detection."""

    encoder_cfg: ClassVar[dict[str, Any]] = {
        "dfine_hgnetv2_n": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_s": {
            "in_channels": [256, 512, 1024],
            "hidden_dim": 256,
            "expansion": 0.5,
            "depth_mult": 0.34,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_m": {
            "in_channels": [384, 768, 1536],
            "hidden_dim": 256,
            "depth_mult": 0.67,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_l": {},
        "dfine_hgnetv2_x": {
            "hidden_dim": 384,
            "dim_feedforward": 2048,
        },
    }

    def __new__(cls, model_name: str) -> HybridEncoderModule:
        """Constructor for HybridEncoder."""
        if model_name not in cls.encoder_cfg:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)
        return HybridEncoderModule(**cls.encoder_cfg[model_name])
