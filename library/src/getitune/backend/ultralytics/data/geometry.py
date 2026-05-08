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
    canvas_size: tuple[int, int] | None = None,
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Convert absolute XYXY bboxes to normalised xywh (Ultralytics format).

    Args:
        bboxes: ``(N, 4)`` tensor or array in ``[x1, y1, x2, y2]`` pixel coords.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        canvas_size: Optional ``(H, W)`` bbox coordinate canvas before tensor resize.
        scale_factor: Optional ``(scale_h, scale_w)`` for letterbox resize.
        padding: Optional ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(N, 4)`` float32 array in ``[cx, cy, w, h]`` normalised by image size.
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    bboxes = bboxes.astype(np.float32).copy()

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if canvas_size is not None and canvas_size != (img_h, img_w):
        bboxes = rescale_bboxes_to_tensor_space(
            bboxes,
            tensor_h=img_h,
            tensor_w=img_w,
            canvas_size=canvas_size,
            scale_factor=scale_factor,
            padding=padding,
        )

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def rescale_bboxes_to_tensor_space(
    bboxes: torch.Tensor | np.ndarray,
    tensor_h: int,
    tensor_w: int,
    canvas_size: tuple[int, int],
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Rescale XYXY bboxes from original/canvas space to tensor pixel space.

    When ``scale_factor`` and ``padding`` are provided (from ``ImageInfo``),
    the transformation accounts for letterbox padding.  Otherwise a simple
    proportional rescale is applied.

    Args:
        bboxes: ``(N, 4)`` tensor or array in ``[x1, y1, x2, y2]`` canvas coords.
        tensor_h: Height of the model input tensor.
        tensor_w: Width of the model input tensor.
        canvas_size: ``(H, W)`` of the coordinate space the bboxes live in.
        scale_factor: ``(scale_h, scale_w)`` applied during resize.
        padding: ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(N, 4)`` float32 array in tensor pixel coordinates.
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    bboxes = bboxes.astype(np.float32).copy()

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if scale_factor is not None:
        scale_h, scale_w = scale_factor
        pad_left, pad_top = padding[0], padding[1]
        bboxes[:, 0::2] = bboxes[:, 0::2] * scale_w + pad_left
        bboxes[:, 1::2] = bboxes[:, 1::2] * scale_h + pad_top
    else:
        canvas_h, canvas_w = canvas_size
        bboxes[:, 0::2] *= tensor_w / canvas_w
        bboxes[:, 1::2] *= tensor_h / canvas_h

    return bboxes


def build_ratio_pad(
    ori_shape: tuple[int, int],
    img_shape: tuple[int, int],
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Derive ``ratio_pad`` from getitune ``ImageInfo`` fields.

    Ultralytics validators need ``ratio_pad = ((rh, rw), (pad_x, pad_y))``
    where ``rh = new_h / ori_h``, ``rw = new_w / ori_w``,
    ``pad_x`` is horizontal (left) padding and ``pad_y`` is vertical (top) padding.

    This matches Ultralytics' ``scale_boxes`` which unpacks as::

        pad_x, pad_y = ratio_pad[1]

    Args:
        ori_shape: ``(H, W)`` of the original image before any transforms.
        img_shape: ``(H, W)`` after resize (before padding).
        padding: ``(left, top, right, bottom)`` padding applied after resize.

    Returns:
        ``((ratio_h, ratio_w), (pad_x, pad_y))`` tuple.
    """
    ori_h, ori_w = ori_shape
    new_h, new_w = img_shape

    # Avoid division-by-zero for degenerate images.
    rh = new_h / ori_h if ori_h > 0 else 1.0
    rw = new_w / ori_w if ori_w > 0 else 1.0

    pad_left, pad_top = padding[0], padding[1]
    return (rh, rw), (pad_left, pad_top)
