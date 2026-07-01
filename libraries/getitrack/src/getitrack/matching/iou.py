# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""IoU computation and Hungarian linear assignment.

All functions operate on plain numpy arrays in ``xyxy`` format. Assignment
uses `scipy.optimize.linear_sum_assignment`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_BBOX_COLS = 4
_EPS = 1e-7
# Sentinel cost for over-threshold pairs, far above any valid cost in [0, 1].
_INVALID_COST = 1e6


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        boxes_a: ``(M, 4)`` float array in ``[x1, y1, x2, y2]`` order.
        boxes_b: ``(N, 4)`` float array in ``[x1, y1, x2, y2]`` order.

    Returns:
        ``(M, N)`` float32 matrix of IoU values in ``[0, 1]``.

    Raises:
        ValueError: If either input is not 2-D with 4 columns, or
            contains non-finite values (NaN, inf).
    """
    _check_shape(boxes_a, "boxes_a")
    _check_shape(boxes_b, "boxes_b")
    _check_finite(boxes_a, "boxes_a")
    _check_finite(boxes_b, "boxes_b")
    m, n = boxes_a.shape[0], boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float32)

    a = boxes_a.astype(np.float32, copy=False)
    b = boxes_b.astype(np.float32, copy=False)
    x1a, y1a, x2a, y2a = a.T
    x1b, y1b, x2b, y2b = b.T

    # Preallocate the four (M, N) buffers we need and reuse the `max`
    # buffers to hold `inter_w` / `inter_h` after the subtraction step.
    x_min_i = np.empty((m, n), dtype=np.float32)
    y_min_i = np.empty_like(x_min_i)
    inter_w = np.empty_like(x_min_i)
    inter_h = np.empty_like(x_min_i)

    np.maximum(x1a[:, None], x1b[None, :], out=x_min_i)
    np.minimum(x2a[:, None], x2b[None, :], out=inter_w)
    np.maximum(y1a[:, None], y1b[None, :], out=y_min_i)
    np.minimum(y2a[:, None], y2b[None, :], out=inter_h)
    np.subtract(inter_w, x_min_i, out=inter_w)
    np.subtract(inter_h, y_min_i, out=inter_h)
    np.clip(inter_w, 0.0, None, out=inter_w)
    np.clip(inter_h, 0.0, None, out=inter_h)

    inter = inter_w * inter_h
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, _EPS)).astype(np.float32)


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU cost (``1 - IoU``) between two sets of boxes.

    Args:
        boxes_a: ``(M, 4)`` ``xyxy`` array.
        boxes_b: ``(N, 4)`` ``xyxy`` array.

    Returns:
        ``(M, N)`` float32 cost matrix in ``[0, 1]``.
    """
    return 1.0 - iou_matrix(boxes_a, boxes_b)


def fuse_score(cost_matrix: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """Weight an IoU cost matrix by per-detection confidence.

    The returned cost is ``1 - (IoU * score)``, so high-confidence
    detections become cheaper to match during assignment.

    Args:
        cost_matrix: ``(M, N)`` IoU cost matrix (``1 - IoU``).
        det_scores: ``(N,)`` detection confidences in ``[0, 1]``.

    Returns:
        ``(M, N)`` fused cost matrix in ``[0, 1]``.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    scores = np.asarray(det_scores, dtype=np.float32)[None, :]
    return (1.0 - iou_sim * scores).astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve Hungarian assignment gated by a maximum-cost threshold.

    Pairs with ``cost > thresh`` are masked to a large sentinel before solving,
    so they cannot displace a valid match in the optimal assignment.

    Args:
        cost_matrix: ``(M, N)`` cost matrix. Rows are typically tracks,
            columns typically detections.
        thresh: Inclusive maximum cost per matched pair.

    Returns:
        A tuple ``(matches, unmatched_rows, unmatched_cols)``:

        - ``matches``: ``(K, 2)`` int array of ``(row, col)`` pairs that
          satisfy the threshold.
        - ``unmatched_rows``: int array of row indices left unmatched.
        - ``unmatched_cols``: int array of column indices left unmatched.
    """
    n_rows, n_cols = cost_matrix.shape
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(n_rows, dtype=np.int64),
            np.arange(n_cols, dtype=np.int64),
        )

    gated = np.where(cost_matrix > thresh, _INVALID_COST, cost_matrix)
    row_ind, col_ind = linear_sum_assignment(gated)
    keep = cost_matrix[row_ind, col_ind] <= thresh
    matched_rows = row_ind[keep]
    matched_cols = col_ind[keep]

    matches = np.stack([matched_rows, matched_cols], axis=1).astype(np.int64)
    matched_row_set = set(matched_rows.tolist())
    matched_col_set = set(matched_cols.tolist())
    unmatched_rows = np.array([i for i in range(n_rows) if i not in matched_row_set], dtype=np.int64)
    unmatched_cols = np.array([j for j in range(n_cols) if j not in matched_col_set], dtype=np.int64)
    return matches, unmatched_rows, unmatched_cols


def _check_shape(arr: np.ndarray, name: str) -> None:
    if arr.ndim != 2 or arr.shape[1] != _BBOX_COLS:
        msg = f"{name} must have shape (N, {_BBOX_COLS}); got {arr.shape}"
        raise ValueError(msg)


def _check_finite(arr: np.ndarray, name: str) -> None:
    if arr.size and not np.isfinite(arr).all():
        msg = f"{name} contains non-finite values (NaN or inf)"
        raise ValueError(msg)
