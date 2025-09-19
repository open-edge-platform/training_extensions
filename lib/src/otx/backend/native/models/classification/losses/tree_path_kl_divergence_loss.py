# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for defining TreePathKLDivergenceLoss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class TreePathKLDivergenceLoss(nn.Module):
    """KL divergence between model distribution over concatenated heads and a target distribution.

    Inputs:
        logits_list: list of tensors [B, C_l], ordered from root -> leaf
        targets: LongTensor [B, L] with per-level GT indices (L == len(logits_list))

    The target distribution places 1/L probability on the GT index for each level,
    and 0 elsewhere, then uses KLDivLoss(log_softmax(logits), target_probs).
    """

    def __init__(self, reduction: str | None = "batchmean", loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kl_div = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, logits_list: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Calculate tree_path KL Divergence loss."""
        if not (isinstance(logits_list, (list, tuple)) and len(logits_list) > 0):
            msg = "logits_list must be non-empty"
            raise ValueError(msg)
        num_levels = len(logits_list)

        # concat logits across all levels
        dims = [t.size(1) for t in logits_list]
        logits_concat = torch.cat(logits_list, dim=1)  # [B, sum(C_l)]
        log_probs = functional.log_softmax(logits_concat, dim=1)  # [B, sum(C_l)]

        # build sparse target distribution with 1/L at each GT index
        batch = log_probs.size(0)
        tgt = torch.zeros_like(log_probs)  # [B, sum(C_l)]
        offset = 0
        for num_c, tgt_l in zip(dims, targets.T):  # level-by-level
            idx_rows = torch.arange(batch, device=log_probs.device)
            tgt[idx_rows, offset + tgt_l] = 1.0 / num_levels
            offset += num_c

        kl = self.kl_div(log_probs, tgt)
        return self.loss_weight * kl
