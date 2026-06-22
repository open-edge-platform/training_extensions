# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Array validation helpers for the core data containers.

Domain-agnostic checks (shape, row alignment, value range, dtype) shared
by the numpy-backed containers in this package.
"""

from __future__ import annotations

import numpy as np

BBOX_COLS = 4


def validate_bboxes(bboxes: np.ndarray) -> None:
    """Raise if ``bboxes`` is not an ``(N, 4)`` array."""
    if bboxes.ndim != 2 or bboxes.shape[1] != BBOX_COLS:
        msg = f"bboxes must have shape (N, {BBOX_COLS}); got {bboxes.shape}"
        raise ValueError(msg)


def validate_row_aligned(*, n: int, **arrays: np.ndarray | None) -> None:
    """Raise if any named array's row count differs from ``n``.

    ``None`` arrays (absent optional fields) are skipped.
    """
    for name, arr in arrays.items():
        if arr is None:
            continue
        if arr.shape[0] != n:
            msg = f"{name} has {arr.shape[0]} rows; expected {n} to match bboxes"
            raise ValueError(msg)


def validate_scores(scores: np.ndarray) -> None:
    """Raise if any score falls outside ``[0, 1]``."""
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        msg = f"scores must be in [0, 1]; got min={scores.min()} max={scores.max()}"
        raise ValueError(msg)


def validate_dtypes(**checks: tuple[np.ndarray, type] | None) -> None:
    """Raise if any ``(array, expected_dtype)`` pair has a mismatched dtype.

    Pass ``None`` for absent optional fields to keep the call site flat.
    """
    for name, item in checks.items():
        if item is None:
            continue
        arr, expected = item
        if arr.dtype != expected:
            msg = f"{name} must have dtype {np.dtype(expected).name}; got {arr.dtype.name}"
            raise TypeError(msg)
