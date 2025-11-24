# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A ruff-friendly, single-file collection of hierarchical classification metrics.

Exports
-------
- :class:`LeafAccuracy` - macro-averaged accuracy at the leaf level.
- :class:`FullPathAccuracy` - exact match across all hierarchy levels.
- :class:`InconsistentPathRatio` - fraction of *predicted* paths violating the tree.
- :class:`WeightedHierarchicalPrecision` - label-count-weighted macro precision over levels.
- :func:`hierMetricCollectionCallable` - returns a ``torchmetrics.MetricCollection`` containing the above metrics.
- :data:`hierMetricCollection` - ``MetricCallable`` alias for integration.

All metrics are compatible with OTX-style :class:`otx.types.label.HLabelInfo`.

"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Precision as TorchPrecision

from otx.metrics.accuracy import HlabelAccuracy
from otx.types.label import HLabelInfo

__all__ = [
    "FullPathAccuracy",
    "HierMetricCollection",
    "InconsistentPathRatio",
    "LeafAccuracy",
    "WeightedHierarchicalPrecision",
    "hier_metric_collection_callable",
]

_INVALID_SHAPE_MSG = "preds and target must have the same shape"
_INVALID_2D_SHAPE = "preds must be 2D (N, L)"


def _build_level_idx_to_name(label_groups: list[list[str]]) -> dict[tuple[int, int], str]:
    """Create a mapping ``(level, index) -> label_name``.

    Args:
        label_groups: ``L`` lists of label names per hierarchy level.
    """
    out: dict[tuple[int, int], str] = {}
    for lvl, labels in enumerate(label_groups):
        for idx, name in enumerate(labels):
            out[(lvl, idx)] = name
    return out


def _make_child_to_parent(edges: list[list[str]]) -> dict[str, str]:
    """Create a mapping ``child -> parent`` from edges."""
    c2p = {}
    for child, parent in edges:
        if child in c2p:  # defensive programming in case of duplicates
            error_msg = f"duplicate child: {child}"
            raise ValueError(error_msg)
        c2p[child] = parent
    return c2p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class LeafAccuracy(Metric):
    """Macro-averaged accuracy at the leaf (last) group.

    Assumes targets/preds are class indices shaped ``(N, L)``.
    """

    full_state_update: bool = False

    def __init__(self, label_info: HLabelInfo) -> None:
        super().__init__()
        self.label_info = label_info

        leaf_labels = label_info.label_groups[-1]
        self.num_leaf_classes = len(leaf_labels)

        self.add_state(
            "correct_per_class",
            default=torch.zeros(self.num_leaf_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_per_class",
            default=torch.zeros(self.num_leaf_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Update state with predictions and targets."""
        pred_leaf = preds[:, -1]
        target_leaf = target[:, -1]
        for cls in range(self.num_leaf_classes):
            mask = target_leaf == cls
            self.total_per_class[cls] += mask.sum()
            self.correct_per_class[cls] += (pred_leaf[mask] == cls).sum()

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """Compute the leaf accuracy metric."""
        total = self.total_per_class.clamp_min_(1)
        per_class_acc = self.correct_per_class.float() / total.float()
        return per_class_acc.mean()


class FullPathAccuracy(Metric):
    """Exact-match accuracy across all hierarchy levels."""

    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Update state with predictions and targets."""
        if preds.shape != target.shape:
            raise ValueError(_INVALID_SHAPE_MSG)
        matches = (preds == target).all(dim=1)
        self.correct += matches.sum()
        self.total += preds.size(0)

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """Compute the full path accuracy metric."""
        return self.correct.float() / self.total.clamp_min(1).float()


class InconsistentPathRatio(Metric):
    """Ratio of *predicted* paths violating the parent→child constraints."""

    full_state_update: bool = False

    def __init__(self, label_info: HLabelInfo) -> None:
        super().__init__()
        self.level_idx_to_name = _build_level_idx_to_name(label_info.label_groups)
        self.child_to_parent = _make_child_to_parent(label_info.label_tree_edges)
        self.add_state("invalid", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Update state with predictions."""
        if preds.ndim != 2:
            raise ValueError(_INVALID_2D_SHAPE)
        n, level = preds.shape
        for i in range(n):
            ok = True
            for lvl in range(1, level):
                child = self.level_idx_to_name[(lvl, int(preds[i, lvl]))]
                parent = self.level_idx_to_name[(lvl - 1, int(preds[i, lvl - 1]))]
                if self.child_to_parent.get(child) != parent:
                    ok = False
                    break
            if not ok:
                self.invalid += 1
        self.total += n

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """Compute the inconsistent path ratio error metric."""
        return self.invalid.float() / self.total.clamp_min(1).float()


class WeightedHierarchicalPrecision(Metric):
    """Label-count-weighted macro precision across hierarchy levels.

    At each level ``l``, computes macro precision and aggregates with weight
    ``|labels_l| / sum_k |labels_k|``. Inputs are class indices ``(N, L)``.
    """

    full_state_update: bool = False

    def __init__(self, label_info: HLabelInfo) -> None:
        super().__init__()
        self.level_sizes: list[int] = []
        self.level_metrics = nn.ModuleList()
        for lvl in sorted(label_info.head_idx_to_logits_range):
            lo, hi = label_info.head_idx_to_logits_range[lvl]
            num_classes = int(hi - lo)
            self.level_sizes.append(num_classes)
            self.level_metrics.append(
                TorchPrecision(task="multiclass", num_classes=num_classes, average="macro"),
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Update state with predictions and targets."""
        # Each column corresponds to a level.
        for lvl, metric in enumerate(self.level_metrics):
            metric.update(preds[:, lvl], target[:, lvl])

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """Compute the wAP."""
        total = float(sum(self.level_sizes))
        weights = [s / total for s in self.level_sizes]
        per_level = [metric.compute() for metric in self.level_metrics]
        return torch.stack([w * v for w, v in zip(weights, per_level)]).sum()

    def reset(self) -> None:  # type: ignore[override]
        """Reset the metric calculation."""
        for metric in self.level_metrics:
            metric.reset()


def hier_metric_collection_callable(label_info: HLabelInfo) -> MetricCollection:
    """Create a ``MetricCollection`` with all hierarchical metrics.

    Returns:
    -------
    torchmetrics.MetricCollection
        Collection with keys: ``leaf_accuracy``, ``full_path_accuracy``,
        ``inconsistent_path_ratio``, ``weighted_precision``.
    """
    return MetricCollection(
        {
            "accuracy": HlabelAccuracy(label_info=label_info),
            "leaf_accuracy": LeafAccuracy(label_info=label_info),
            "full_path_accuracy": FullPathAccuracy(),
            "inconsistent_path_ratio": InconsistentPathRatio(label_info=label_info),
            "weighted_precision": WeightedHierarchicalPrecision(label_info=label_info),
        },
    )


HMetricCallable = Callable[[HLabelInfo], Metric | MetricCollection]

HierMetricCollection: HMetricCallable = hier_metric_collection_callable
