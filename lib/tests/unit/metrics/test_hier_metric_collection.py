# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import re
import types

import pytest
import torch
from torchmetrics.classification import Precision as TorchPrecision

from otx.metrics.hier_metric_collection import (
    FullPathAccuracy,
    InconsistentPathRatio,
    LeafAccuracy,
    WeightedHierarchicalPrecision,
    hier_metric_collection_callable,
)


@pytest.fixture()
def label_info_stub():
    """Minimal stub that mimics the LabelInfo attributes used by our metrics.

    - label_groups: list[list[str]] per hierarchy level
    - label_tree_edges: list[tuple[str, str]] as (child, parent)
    - head_idx_to_logits_range: dict[level, (lo, hi)] for per-level class counts
    """
    li = types.SimpleNamespace()
    li.label_groups = [
        ["Boeing", "Airbus"],
        ["737", "A320"],
        ["737-800", "737-900", "A320-200", "A320-neo"],
    ]
    # edges are (child, parent)
    li.label_tree_edges = [
        ("737", "Boeing"),
        ("A320", "Airbus"),
        ("737-800", "737"),
        ("737-900", "737"),
        ("A320-200", "A320"),
        ("A320-neo", "A320"),
    ]
    # per-level class ranges (concatenated logits indices convention)
    li.head_idx_to_logits_range = {0: (0, 2), 1: (2, 4), 2: (4, 8)}
    li.num_multiclass_heads = len(li.label_groups)
    li.num_multilabel_classes = sum(len(g) for g in li.label_groups)
    return li


@pytest.fixture()
def sample_tensors():
    """Return (target, preds) shaped (N, L) with class indices.

    N=4, L=3 (3 hierarchy levels)
    """
    # targets (true) indices per level
    target = torch.tensor(
        [
            [0, 0, 0],  # Boeing, 737, 737-800
            [1, 1, 2],  # Airbus, A320, A320-200
            [0, 0, 1],  # Boeing, 737, 737-900
            [1, 1, 3],  # Airbus, A320, A320-neo
        ]
    )
    # preds: 2 exact matches (rows 1 and 2); two leaf errors
    preds = torch.tensor(
        [
            [0, 0, 1],  # leaf wrong
            [1, 1, 2],  # exact
            [0, 0, 1],  # exact
            [1, 1, 0],  # leaf wrong
        ]
    )
    return target, preds


# ------------------------------ LeafAccuracy --------------------------------


def test_leaf_accuracy_macro_mean(label_info_stub, sample_tensors):
    target, preds = sample_tensors
    metric = LeafAccuracy(label_info_stub)
    metric.update(preds, target)
    val = metric.compute().item()

    # Compute expected macro mean at leaf by hand
    y_true_leaf = target[:, -1]
    y_pred_leaf = preds[:, -1]
    per_class = []
    for cls in range(4):
        mask = y_true_leaf == cls
        tot = int(mask.sum())
        if tot == 0:
            per_class.append(0.0)
        else:
            correct = int((y_pred_leaf[mask] == cls).sum())
            per_class.append(correct / tot)
    expected = sum(per_class) / 4.0
    assert val == pytest.approx(expected)


# --------------------------- FullPathAccuracy -------------------------------
def test_full_path_accuracy(sample_tensors):
    target, preds = sample_tensors
    metric = FullPathAccuracy()
    metric.update(preds, target)
    val = metric.compute().item()
    # exact rows: 1 and 2 -> 2/4
    assert val == pytest.approx(0.5)


# ------------------------ Inconsistent Path Ratio ---------------------------
def test_inconsistent_path_ratio_inconsistent(label_info_stub):
    # Make structurally invalid predictions (wrong parent chain)
    preds_bad = torch.tensor(
        [
            [0, 1, 0],  # 737 belongs to Boeing, not Airbus
            [1, 0, 3],  # A320 belongs to Airbus, not Boeing
        ]
    )
    target_dummy = torch.zeros_like(preds_bad)

    metric = InconsistentPathRatio(label_info_stub)
    metric.update(preds_bad, target_dummy)
    val = metric.compute().item()
    assert val == pytest.approx(1.0)


# --------------------- Weighted Hierarchical Precision ----------------------
def test_weighted_hierarchical_precision_matches_reference(label_info_stub, sample_tensors):
    target, preds = sample_tensors
    metric = WeightedHierarchicalPrecision(label_info_stub)
    metric.update(preds, target)
    got = metric.compute().item()

    # Reference computation using TorchPrecision per level, then label-count weights
    level_sizes = [len(g) for g in label_info_stub.label_groups]
    total = float(sum(level_sizes))
    weights = [s / total for s in level_sizes]

    ref_vals = []
    for lvl, ncls in enumerate(level_sizes):
        ref = TorchPrecision(task="multiclass", num_classes=ncls, average="macro")
        ref.update(preds[:, lvl], target[:, lvl])
        ref_vals.append(ref.compute().item())

    expected = sum(w * v for w, v in zip(weights, ref_vals))
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


# -------------------------- MetricCollection callable -----------------------
def test_hier_metric_collection_callable(label_info_stub, sample_tensors):
    target, preds = sample_tensors
    mc = hier_metric_collection_callable(label_info_stub)

    # update/compute over the whole collection
    mc.update(preds, target)
    out = mc.compute()

    assert set(out.keys()) == {
        "leaf_accuracy",
        "full_path_accuracy",
        "inconsistent_path_ratio",
        "weighted_precision",
        "accuracy",
        "conf_matrix",
    }

    # spot-check a couple of values
    assert out["full_path_accuracy"].item() == pytest.approx(0.5)
    assert out["inconsistent_path_ratio"].item() == pytest.approx(0.25)


# ------------------------------ Error handling ------------------------------


def test_full_path_accuracy_shape_mismatch_raises():
    metric = FullPathAccuracy()
    preds = torch.tensor([[0, 0, 0]])
    target = torch.tensor([[0, 0]])  # wrong shape
    with pytest.raises(ValueError, match=re.escape("preds and target must have the same shape")):
        metric.update(preds, target)


def test_inconsistent_path_ratio_requires_2d(label_info_stub):
    metric = InconsistentPathRatio(label_info_stub)
    preds = torch.tensor([0, 1, 2])  # 1D
    target = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match=re.escape("preds must be 2D (N, L)")):
        metric.update(preds, target)
