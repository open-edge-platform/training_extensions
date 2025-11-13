# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

from otx.backend.native.models.classification.losses.tree_path_kl_divergence_loss import TreePathKLDivergenceLoss


@pytest.mark.parametrize(
    ("levels", "classes_per_level"),
    [
        (2, [3, 5]),
        (3, [2, 3, 4]),
    ],
)
def test_forward_scalar_and_finite(levels, classes_per_level):
    torch.manual_seed(0)
    batch = 4
    logits_list = [torch.randn(batch, c) for c in classes_per_level]
    targets = torch.stack([torch.randint(0, c, (batch,)) for c in classes_per_level], dim=1)

    loss_fn = TreePathKLDivergenceLoss(reduction="batchmean")
    loss = loss_fn(logits_list, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-7


def test_backward_produces_grads():
    batch = 3
    channel = [4, 6]
    logits_list = [torch.randn(batch, c, requires_grad=True) for c in channel]
    targets = torch.stack([torch.randint(0, c, (batch,)) for c in channel], dim=1)

    loss = TreePathKLDivergenceLoss()(logits_list, targets)
    loss.backward()
    for logit in logits_list:
        assert logit.grad is not None
        assert torch.isfinite(logit.grad).all()


def test_alignment_vs_misalignment_loss():
    batch = 2
    channel0, channel1 = 3, 4
    targets = torch.tensor([[0, 1], [2, 3]])

    # Aligned: boost GT logits
    aligned0 = torch.zeros(batch, channel0)
    aligned1 = torch.zeros(batch, channel1)
    aligned0[torch.arange(batch), targets[:, 0]] = 5.0
    aligned1[torch.arange(batch), targets[:, 1]] = 5.0

    # Misaligned: boost wrong logits
    mis0 = torch.zeros(batch, channel0)
    mis1 = torch.zeros(batch, channel1)
    mis0[torch.arange(batch), (targets[:, 0] + 1) % channel0] = 5.0
    mis1[torch.arange(batch), (targets[:, 1] + 1) % channel1] = 5.0

    loss_fn = TreePathKLDivergenceLoss()
    loss_aligned = loss_fn([aligned0, aligned1], targets)
    loss_misaligned = loss_fn([mis0, mis1], targets)
    assert loss_aligned < loss_misaligned


def test_single_level_exact_value():
    """
    With a single level, KL reduces to CE between predicted softmax and one-hot target.
    We check exact value against F.cross_entropy.
    """

    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]])
    targets = torch.tensor([[0], [2]])  # shape [B,1]

    # TreePathKLP
    loss_fn = TreePathKLDivergenceLoss(reduction="batchmean")
    kl_loss = loss_fn([logits], targets)

    # CrossEntropy with one-hot is same as NLLLoss(log_softmax)
    ce_loss = functional.cross_entropy(logits, targets.view(-1), reduction="mean")

    assert torch.allclose(kl_loss, ce_loss, atol=1e-6)


def test_multi_level_exact_value_batchmean():
    """
    Exact numerical check for L=2 levels with 'batchmean' reduction.

    Loss per sample (PyTorch KLDivLoss):
      KL(p || q) = sum_j p_j * (log(p_j) - log(q_j))
    where input to KLDivLoss is log(q_j) (our model log_probs),
    and the target is p_j (our constructed target distribution).
    With reduction='batchmean', PyTorch divides the total sum by batch size.
    """

    # Use double for better numerical agreement
    batch = 2
    l0, l1 = 2, 3
    logits0 = torch.tensor([[2.0, -1.0], [0.0, 1.0]], dtype=torch.float64)  # [B, l0]
    logits1 = torch.tensor([[0.5, 0.0, -0.5], [-1.0, 2.0, 0.5]], dtype=torch.float64)  # [B, l1]

    # Ground-truth indices per level
    # sample 0: level0->0, level1->1
    # sample 1: level0->1, level1->2
    targets = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # [B, 2]
    level = 2  # number of levels

    # Model log probs over concatenated heads
    concat = torch.cat([logits0, logits1], dim=1)  # [B, l0+l1]
    log_q = functional.log_softmax(concat, dim=1)  # log(q_j)

    # Build target distribution p: 1/level at each GT index, 0 elsewhere
    p = torch.zeros_like(log_q, dtype=torch.float64)
    offset = 0
    for num_c, tgt_l in zip([l0, l1], targets.T):
        rows = torch.arange(batch)
        p[rows, offset + tgt_l] = 1.0 / level
        offset += num_c

    # Manual KL with 'batchmean' reduction:
    # sum_i sum_j p_ij * (log p_ij - log q_ij) / batch
    # (avoid log(0) by masking since p is sparse)
    mask = p > 0
    log_p = torch.zeros_like(p)
    log_p[mask] = torch.log(p[mask])
    manual_kl = (p * (log_p - log_q)).sum() / batch

    # Loss under test (must match manual)
    loss_fn = TreePathKLDivergenceLoss(reduction="batchmean")
    test_kl = loss_fn([logits0.float(), logits1.float()], targets)

    assert torch.allclose(test_kl.double(), manual_kl, atol=1e-8), (
        f"manual={manual_kl.item():.12f} vs loss={test_kl.item():.12f}"
    )
