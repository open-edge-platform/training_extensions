# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Implementation of common transformer layers."""

from __future__ import annotations

import copy
import math
from typing import Callable, List, Tuple
import functools

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.nn import init

from otx.backend.native.models.common.utils.utils import get_clones
from otx.backend.native.models.modules.transformer import deformable_attention_core_func, deformable_attention_core_func_v2


def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalize_before: bool = False,
        batch_first: bool = True,
        key_mask: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.key_mask = key_mask

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        """Attach position embeddings to the tensor."""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder layer.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if self.key_mask:
            src, _ = self.self_attn(q, k, value=src, key_padding_mask=src_mask, need_weights=False)
        else:
            src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerEncoder(nn.Module):
    """TransformerEncoder."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        """Initialize the TransformerEncoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of layers.
            norm (nn.Module | None, optional): The normalization module. Defaults to None.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MLP(nn.Module):
    """A classic Multi Layer Perceptron (MLP).

    Args:
        input_dim (int): The number of expected features in the input.
        hidden_dim (int): The number of features in the hidden layers.
        output_dim (int): The number of features in the output layer.
        num_layers (int): The number of layers in the MLP.
        activation (Callable[..., nn.Module] | None, optional): The activation function. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.act = activation() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of MLP."""
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        num_levels (int): The number of levels in MSDeformableAttention.
        num_points (int): The number of points in MSDeformableAttention.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4) -> None:
        """Multi-Scale Deformable Attention Module."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            msg = f"embed_dim must be divisible by num_heads, but got embed_dim={embed_dim} and num_heads={num_heads}"
            raise ValueError(msg)

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of MSDeformableAttention.

        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (list[tuple[int, int]]): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ...]
            value_mask (Tensor | None, optional): [bs, value_length], True for non-padding elements,
                False for padding elements. Defaults to None.

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = nn.functional.softmax(attention_weights, dim=-1).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = (
                value_spatial_shapes
                if isinstance(value_spatial_shapes, torch.Tensor)
                else torch.tensor(value_spatial_shapes)
            ).clone()
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(
                    bs,
                    len_q,
                    1,
                    self.num_levels,
                    1,
                    2,
                )
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        elif reference_points.shape[-1] == 6:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * (reference_points[:, :, None, :, None, 2::2] + reference_points[:, :, None, :, None, 3::2])
                * 0.5
            )
        else:
            msg = f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            raise ValueError(
                msg,
            )

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        return self.output_proj(output)


class MSDeformableAttentionV2(nn.Module):
    """Multi-Scale Deformable Attention Module V2.

    Note:
        This is different from vanilla MSDeformableAttention where it uses
        distinct number of sampling points for features at different scales.
        Refer to RTDETRv2.

    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        num_levels (int): The number of levels in MSDeformableAttention.
        num_points_list (list[int]): Number of distinct points for each layer. Defaults to [3, 6, 3].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points_list: list[int] = [3, 6, 3],  # noqa: B006
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale",
            torch.tensor(num_points_scale, dtype=torch.float32),
        )

        self.total_points = num_heads * sum(num_points_list)
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters of the model."""
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        value_spatial_shapes: list[list[int]],
    ) -> Tensor:
        """Forward function of MSDeformableAttention.

        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        _, n_head, c, _ = value[0].shape
        num_points_list = self.num_points_list

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs,
            len_q,
            self.num_heads,
            sum(self.num_points_list),
            2,
        )

        attention_weights = self.attention_weights(query).reshape(
            bs,
            len_q,
            self.num_heads,
            sum(self.num_points_list),
        )
        attention_weights = f.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(
                    bs,
                    len_q,
                    1,
                    self.num_levels,
                    1,
                    2,
                )
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(query).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * 0.5
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            msg = (f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.",)
            raise ValueError(msg)

        # sampling_offsets [8, 480, 8, 12, 2]
        sampling_grids = 2 * sampling_locations - 1

        sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            value_l = value[level].reshape(bs * n_head, c, h, w)
            sampling_grid_l = sampling_locations_list[level]
            sampling_value_l = f.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampling_value_list.append(sampling_value_l)

        attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
        weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
        output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, len_q)

        return output.permute(0, 2, 1)


class MLPV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttentionV3(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttentionV3, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = f.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class VisualEncoderLayer(nn.Module):
    """VisualEncoderLayer module consisting of MSDeformableAttention and feed-forward network.

    Args:
        d_model (int): The input and output dimension of the layer. Defaults to 256.
        d_ffn (int): The hidden dimension of the feed-forward network. Defaults to 1024.
        dropout (float): The dropout rate. Defaults to 0.1.
        activation (Callable[..., nn.Module]): The activation function. Defaults to nn.ReLU.
        n_levels (int): The number of feature levels. Defaults to 4.
        n_heads (int): The number of attention heads. Defaults to 8.
        n_points (int): The number of sampling points for the MSDeformableAttention. Defaults to 4.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()

        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add position embedding to the input tensor.

        Args:
            tensor (Tensor): The input tensor.
            pos (Tensor | None): The position embedding tensor. Defaults to None.

        Returns:
            Tensor: The tensor with position embedding added.
        """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        """Forward pass of the ffn.

        Args:
            src (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the VisualEncoderLayer.

        Args:
            src (Tensor): The input tensor.
            pos (Tensor): The position embedding tensor.
            reference_points (Tensor): The reference points tensor.
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            padding_mask (Optional[Tensor]): The padding mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        return self.forward_ffn(src)


class VisualEncoder(nn.Module):
    """VisualEncoder module consisting of multiple VisualEncoderLayer modules.

    Args:
        encoder_layer (VisualEncoderLayer): The Visual encoder layer module.
        num_layers (int): The number of layers.
    """

    def __init__(self, encoder_layer: VisualEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Generate reference points for each spatial level.

        Args:
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            valid_ratios (Tensor): The tensor of valid ratios.
            device (torch.device): The device to use.

        Returns:
            Tensor: The tensor of reference points.
        """
        reference_points_list = []
        for lvl, (h_, w_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_ - 0.5, h_, device=device),
                torch.linspace(0.5, w_ - 0.5, w_, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the VisualEncoder module.

        Args:
            src (Tensor): The input tensor.
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            level_start_index (Tensor): The level start index tensor.
            valid_ratios (Tensor): The tensor of valid ratios.
            pos (Tensor | None): The position embedding tensor. Defaults to None.
            padding_mask (Tensor | None): The padding mask tensor. Defaults to None.
            ref_token_index (int | None): The reference token index. Defaults to None.
            ref_token_coord (Tensor | None): The reference token coordinates. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output

######### TODO(kprokofi): Remove duplicates #########


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped

# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return f.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError
