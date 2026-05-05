# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for transformers 5.x breaking changes.

In transformers 5.0, several utilities were removed or relocated.  The ``rfdetr``
package still imports them from the old locations, so we re-inject vendored copies
before rfdetr is loaded.

Patched symbols:
- ``transformers.pytorch_utils.find_pruneable_heads_and_indices``
- ``transformers.utils.backbone_utils.get_aligned_output_features_output_indices``
- ``transformers.backbone_utils.BackboneMixin._init_backbone``
"""

from __future__ import annotations

import torch
from transformers import pytorch_utils
from transformers.backbone_utils import BackboneMixin
from transformers.utils import backbone_utils


def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """Find the heads and their indices taking ``already_pruned_heads`` into account."""
    mask = torch.ones(n_heads, head_size)
    heads_set = set(heads) - already_pruned_heads
    for head in heads_set:
        adjusted = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[adjusted] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads_set, index


def get_aligned_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: list[int] | tuple[int, ...] | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    """Align ``out_features`` and ``out_indices`` based on ``stage_names``."""
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
    return [stage_names[-1]], [len(stage_names) - 1]


def _init_backbone(self, config) -> None:  # noqa: ANN001
    """Re-implementation of BackboneMixin._init_backbone removed in transformers 5.0."""
    self.stage_names = config.stage_names
    self._out_features = config._out_features  # noqa: SLF001
    self._out_indices = config._out_indices  # noqa: SLF001


if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
    # pyrefly: ignore[missing-attribute]
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(backbone_utils, "get_aligned_output_features_output_indices"):
    # pyrefly: ignore[missing-attribute]
    backbone_utils.get_aligned_output_features_output_indices = get_aligned_output_features_output_indices

if not hasattr(BackboneMixin, "_init_backbone"):
    BackboneMixin._init_backbone = _init_backbone  # type: ignore[attr-defined]  # noqa: SLF001
