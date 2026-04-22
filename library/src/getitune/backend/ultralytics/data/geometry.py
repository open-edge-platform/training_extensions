# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geometry and bbox conversion utilities for Ultralytics data bridge."""

from __future__ import annotations

import numpy as np
import torch


def xyxy_abs_to_xywh_norm(
    bboxes: torch.Tensor | np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert absolute XYXY bboxes to normalised xywh (Ultralytics format).

    Args:
        bboxes: ``(N, 4)`` tensor or array in ``[x1, y1, x2, y2]`` pixel coords.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        ``(N, 4)`` float32 array in ``[cx, cy, w, h]`` normalised by image size.
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    bboxes = bboxes.astype(np.float32).copy()

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def build_ratio_pad(
    ori_shape: tuple[int, int],
    img_shape: tuple[int, int],
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Derive ``ratio_pad`` from getitune ``ImageInfo`` fields.

    Ultralytics validators need ``ratio_pad = ((rh, rw), (pad_top, pad_left))``
    where ``rh = new_h / ori_h`` and ``rw = new_w / ori_w``.

    Args:
        ori_shape: ``(H, W)`` of the original image before any transforms.
        img_shape: ``(H, W)`` after resize (before padding).
        padding: ``(left, top, right, bottom)`` padding applied after resize.

    Returns:
        ``((ratio_h, ratio_w), (pad_top, pad_left))`` tuple.
    """
    ori_h, ori_w = ori_shape
    new_h, new_w = img_shape

    # Avoid division-by-zero for degenerate images.
    rh = new_h / ori_h if ori_h > 0 else 1.0
    rw = new_w / ori_w if ori_w > 0 else 1.0

    pad_left, pad_top = padding[0], padding[1]
    return (rh, rw), (pad_top, pad_left)
