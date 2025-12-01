# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Common components shared between D-FINE and DEIM transformer decoders."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.nn import init
import torchvision

from otx.backend.native.models.common.layers.transformer_layers import MLP
from otx.backend.native.models.utils.weight_init import bias_init_with_prob
from otx.backend.native.models.common.utils.utils import inverse_sigmoid


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim (int): The number of features in the input.
        eps (float, optional): A value added for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of RMSNorm.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.scale

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"dim={self.dim}, eps={self.eps}"

    def reset_parameters(self) -> None:
        nn.init.constant_(self.scale, 1)


class Gate(nn.Module):
    """Target Gating Layer with learnable fusion weights.

    This module combines two input tensors using learnable gating weights,
    allowing the model to dynamically control information flow.

    Args:
        d_model (int): The number of expected features in the input.
        use_rmsnorm (bool, optional): Whether to use RMSNorm instead of LayerNorm. Defaults to False.
    """

    def __init__(self, d_model: int, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass of gating mechanism.

        Args:
            x1 (Tensor): First input tensor.
            x2 (Tensor): Second input tensor.

        Returns:
            Tensor: Gated and normalized output tensor.
        """
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """Convert distribution predictions to continuous bounding box coordinates.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int, optional): Maximum number of discrete bins. Defaults to 32.
    """

    def __init__(self, reg_max: int = 32) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: Tensor, box_distance_weight: Tensor) -> Tensor:
        """Convert distribution to coordinates.

        Args:
            x (Tensor): Distribution predictions of shape (..., 4*(reg_max+1)).
            box_distance_weight (Tensor): Weighting function for integration.

        Returns:
            Tensor: Continuous bounding box coordinates of shape (..., 4).
        """
        shape = x.shape
        x = f.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = f.linear(x, box_distance_weight).reshape(-1, 4)
        return x.reshape([*list(shape[:-1]), -1])


class LQE(nn.Module):
    """Localization Quality Estimation module.

    Estimates the quality of predicted bounding boxes by analyzing the
    distribution statistics of corner predictions.

    Args:
        k (int): Number of top-k edge points to consider.
        hidden_dim (int): Hidden dimension for the MLP.
        num_layers (int): Number of MLP layers.
        reg_max (int): Maximum number of discrete bins for bbox regression.
        act (Callable[..., nn.Module], optional): Activation function. Defaults to ReLU.
    """

    def __init__(
        self,
        k: int,
        hidden_dim: int,
        num_layers: int,
        reg_max: int,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(
            input_dim=4 * (k + 1),
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation=activation,
        )
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: Tensor, pred_corners: Tensor) -> Tensor:
        """Estimate localization quality and adjust scores.

        Args:
            scores (Tensor): Predicted classification scores of shape (B, N, C).
            pred_corners (Tensor): Predicted bounding box corners of shape (B, N, 4*(reg_max+1)).

        Returns:
            Tensor: Quality-adjusted scores of shape (B, N, C).
        """
        b, num_pred, _ = pred_corners.size()
        prob = f.softmax(pred_corners.reshape(b, num_pred, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(b, num_pred, -1))
        return scores + quality_score


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Initializes SwiGLUFFN module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLUFFN module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = f.silu(x1) * x2
        return self.w3(hidden)


def get_contrastive_denoising_training_group(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: torch.nn.Module,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[None, None, None, None]:
    """Generate contrastive denoising training group.

    Args:
        targets (List[Dict[str, torch.Tensor]]): List of target dictionaries.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.nn.Module): Class embedding module.
        num_denoising (int, optional): Number of denoising queries. Defaults to 100.
        label_noise_ratio (float, optional): Ratio of label noise. Defaults to 0.5.
        box_noise_scale (float, optional): Scale of box noise. Defaults to 1.0.

    Returns:
        Tuple[Tensor,Tensor,Tensor, dict[str, Tensor]] | tuple[None,None,None,None]:
        Tuple containing input query class, input query bbox, attention mask, and denoising metadata.
    """
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = torchvision.ops.box_convert(input_query_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = torchvision.ops.box_convert(known_bbox, in_fmt="xyxy", out_fmt="cxcywh")
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta
