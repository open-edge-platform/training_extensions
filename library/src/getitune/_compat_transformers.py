# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for transformers 5.x breaking changes.

In transformers 5.0, several utilities were removed or relocated.  The ``rfdetr``
package still imports them from the old locations, so we re-inject vendored copies
before rfdetr is loaded.

Patched symbols:
- ``transformers.pytorch_utils.find_pruneable_heads_and_indices``
- ``transformers.utils.backbone_utils.get_aligned_output_features_output_indices``
"""

from __future__ import annotations

import torch
from transformers import pytorch_utils
from transformers.utils import backbone_utils

# ---------------------------------------------------------------------------
# find_pruneable_heads_and_indices (removed in transformers 5.0)
# ---------------------------------------------------------------------------


def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """Find the heads and their indices taking ``already_pruned_heads`` into account.

    Args:
        heads: List of the indices of heads to prune.
        n_heads: The number of heads in the model.
        head_size: The size of each head.
        already_pruned_heads: A set of already pruned heads.

    Returns:
        A tuple with the remaining heads to prune and the corresponding weight indices to keep.
    """
    mask = torch.ones(n_heads, head_size)
    heads_set = set(heads) - already_pruned_heads
    for head in heads_set:
        adjusted = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[adjusted] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads_set, index


# ---------------------------------------------------------------------------
# get_aligned_output_features_output_indices (removed in transformers 5.0)
# ---------------------------------------------------------------------------


def get_aligned_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: list[int] | tuple[int, ...] | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    """Align ``out_features`` and ``out_indices`` based on ``stage_names``.

    Args:
        out_features: List of feature names to output.
        out_indices: List of stage indices to output.
        stage_names: Ordered list of all stage names in the model.

    Returns:
        A consistent (out_features, out_indices) tuple.
    """
    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            msg = "out_features and out_indices should have the same length if both are set"
            raise ValueError(msg)
        if out_features != [stage_names[idx] for idx in out_indices]:
            msg = "out_features and out_indices should correspond to the same stages if both are set"
            raise ValueError(msg)
        return out_features, list(out_indices)
    if out_features is not None:
        out_indices_resolved = [stage_names.index(layer) for layer in out_features]
        return out_features, out_indices_resolved
    if out_indices is not None:
        out_features_resolved = [stage_names[idx] for idx in out_indices]
        return out_features_resolved, list(out_indices)
    # Default: last stage only
    return [stage_names[-1]], [len(stage_names) - 1]


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
    # pyrefly: ignore[missing-attribute]
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(backbone_utils, "get_aligned_output_features_output_indices"):
    # pyrefly: ignore[missing-attribute]
    backbone_utils.get_aligned_output_features_output_indices = get_aligned_output_features_output_indices
