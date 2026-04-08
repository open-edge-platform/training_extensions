# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom image transforms for OTX augmentation pipeline."""

from __future__ import annotations

import copy
import math
import typing
from typing import Any, cast

import torch
import torchvision.transforms.v2 as tvt_v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812

from otx.data.augmentation.kernels import (
    _resize_image_info,
    _resized_crop_image_info,
)
from otx.data.entity.sample import OTXSample


class Resize(tvt_v2.Transform):
    """Resize transform based on torchvision.transforms.v2.

    Extends torchvision's Resize with optional control over target resizing
    and aspect ratio preservation with padding.

    Args:
        size (int or tuple): Target size (height, width). If int, a square (size, size) is used.
        resize_targets (bool): If True, resize all targets (bboxes, masks, keypoints)
            along with the image. If False, resize only the image and leave targets unchanged.
            Defaults to True.
        keep_aspect_ratio (bool): If True, preserve the aspect ratio of the original image
            by resizing to fit within the target size and padding to reach exact target dimensions.
            Defaults to False.
        pad_value (int or tuple): Padding value for image. Defaults to 0.
        interpolation: Interpolation mode for images. Defaults to InterpolationMode.BILINEAR.
        antialias: Whether to apply antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        resize_targets: bool = True,
        keep_aspect_ratio: bool = False,
        pad_value: int | tuple[int, int, int] = 0,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.resize_targets = resize_targets
        self.keep_aspect_ratio = keep_aspect_ratio
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.antialias = antialias

    def _compute_resize_params(
        self,
        orig_h: int,
        orig_w: int,
    ) -> tuple[int, int, int, int, int, int]:
        """Compute resize dimensions and padding.

        Returns:
            new_h, new_w: Size after resize (before padding).
            pad_left, pad_top, pad_right, pad_bottom: Padding amounts.
        """
        target_h, target_w = self.size

        if not self.keep_aspect_ratio:
            return target_h, target_w, 0, 0, 0, 0

        # Compute scale to fit within target while preserving aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        # Compute padding to reach target size
        # Use bottom-right padding only (matching develop branch behavior)
        # This is important because model post-processing assumes no left/top offset
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = 0
        pad_right = pad_w
        pad_top = 0
        pad_bottom = pad_h

        return new_h, new_w, pad_left, pad_top, pad_right, pad_bottom

    def forward(self, *inputs: OTXSample) -> OTXSample:  # type: ignore[override]
        """Resize image and optionally targets, with optional aspect ratio preservation."""
        if len(inputs) > 1:
            msg = "Resize expects a single OTXSample input"
            raise ValueError(msg)
        sample: OTXSample = inputs[0]

        if not hasattr(sample, "image"):
            # Fallback: just resize the tensor directly
            return cast(
                "OTXSample",
                F.resize(
                    cast("torch.Tensor", sample),
                    size=list(self.size),
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                ),
            )

        # Get original dimensions
        orig_h, orig_w = sample.image.shape[-2:]

        # Compute resize and padding parameters
        new_h, new_w, pad_left, pad_top, pad_right, pad_bottom = self._compute_resize_params(orig_h, orig_w)

        # Resize image
        sample.image = F.resize(
            sample.image,
            size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        sample.image = sample.image.clamp(0, 1)
        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            fill_value: float | int | list[float] = (
                list(self.pad_value) if isinstance(self.pad_value, tuple) else self.pad_value
            )
            sample.image = F.pad(
                sample.image,
                padding=[pad_left, pad_top, pad_right, pad_bottom],
                fill=fill_value,
            )

        # Calculate scale factors for target transforms
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # Resize/transform targets if requested
        if self.resize_targets:
            # Transform bounding boxes
            bboxes = getattr(sample, "bboxes", None)
            if bboxes is not None and len(bboxes) > 0:
                # Scale bboxes
                if isinstance(bboxes, tv_tensors.BoundingBoxes):
                    bboxes_data = bboxes.clone()
                else:
                    bboxes_data = bboxes.clone() if isinstance(bboxes, torch.Tensor) else torch.as_tensor(bboxes)

                # Apply scaling
                bboxes_data[..., 0::2] = bboxes_data[..., 0::2] * scale_x  # x coordinates
                bboxes_data[..., 1::2] = bboxes_data[..., 1::2] * scale_y  # y coordinates

                # Apply padding offset
                if pad_left > 0 or pad_top > 0:
                    bboxes_data[..., 0::2] = bboxes_data[..., 0::2] + pad_left  # x coordinates
                    bboxes_data[..., 1::2] = bboxes_data[..., 1::2] + pad_top  # y coordinates

                sample.bboxes = tv_tensors.BoundingBoxes(  # type: ignore[missing-attribute]
                    bboxes_data,
                    format=bboxes.format if isinstance(bboxes, tv_tensors.BoundingBoxes) else "XYXY",
                    canvas_size=self.size,
                )

            # Transform masks
            masks = getattr(sample, "masks", None)
            if masks is not None and len(masks) > 0:
                # Resize masks
                resized_masks = F.resize(
                    masks,
                    size=[new_h, new_w],
                    interpolation=F.InterpolationMode.NEAREST,
                    antialias=False,
                )
                # Pad masks
                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    resized_masks = F.pad(
                        resized_masks,
                        padding=[pad_left, pad_top, pad_right, pad_bottom],
                        fill=0,
                    )
                sample.masks = (  # type: ignore[missing-attribute]
                    tv_tensors.Mask(resized_masks) if isinstance(masks, tv_tensors.Mask) else resized_masks
                )

            # Transform keypoints/points
            keypoints = getattr(sample, "keypoints", None)
            if keypoints is not None and isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.clone()
                # Scale keypoints (assuming format [..., x, y] or [..., x, y, visibility])
                keypoints[..., 0] = keypoints[..., 0] * scale_x + pad_left
                keypoints[..., 1] = keypoints[..., 1] * scale_y + pad_top
                sample.keypoints = keypoints  # type: ignore[missing-attribute]

        # Update img_info if available
        if hasattr(sample, "img_info") and sample.img_info is not None:
            # First update scale info based on resized (pre-pad) shape
            sample.img_info = _resize_image_info(sample.img_info, (new_h, new_w))
            # Then apply padding metadata if any
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                sample.img_info.padding = (pad_left, pad_top, pad_right, pad_bottom)
                sample.img_info.img_shape = (
                    new_h + pad_top + pad_bottom,
                    new_w + pad_left + pad_right,
                )
            if self.keep_aspect_ratio:
                # Store padding info for potential inverse transforms
                sample.img_info.pad_offset = (pad_left, pad_top, pad_right, pad_bottom)  # type: ignore[missing-attribute]
                # ImageInfo.scale_factor uses (height, width)
                sample.img_info.scale_factor = (scale_y, scale_x)
                sample.img_info.keep_ratio = True

        return sample


class CachedMosaic(tvt_v2.Transform):
    """Mosaic augmentation with caching and built-in affine crop.

    Combines four images into a 2× mosaic canvas, then applies random affine
    distortion and crops back to ``img_scale``.  This matches the standard
    YOLOX pipeline (mmdetection / Ultralytics) where ``Mosaic`` and
    ``RandomAffine(border=-img_scale/2)`` form a single logical step.

    When the mosaic is **skipped** (probability gate or insufficient cache) the
    input is resized to ``img_scale`` so the output size is always fixed.

    Uses only torch/torchvision operations — no numpy/cv2.

    Args:
        img_scale (tuple[int, int]): Per-tile target size ``(H, W)``; also the
            **output** size after affine crop.  Defaults to (640, 640).
        center_ratio_range (tuple[float, float]): Range for the random mosaic
            centre as a ratio of ``img_scale``.  Defaults to (0.5, 1.5).
        bbox_clip_border (bool): Clip bboxes to the canvas boundary.
            Defaults to True.
        pad_val (float | tuple[float, float, float]): Fill value for the mosaic
            canvas and affine OOB regions ([0, 255] scale; normalised
            automatically).  Defaults to 114.0.
        p (float): Probability of applying mosaic. Defaults to 1.0.
        max_cached_images (int): Maximum cache size (>= 4). Defaults to 40.
        random_pop (bool): Random eviction when cache is full; FIFO otherwise.
            Defaults to True.
        max_rotate_degree (float): Max rotation for affine crop (degrees).
            Defaults to 10.0.
        scaling_ratio_range (tuple[float, float]): Scale jitter ``(min, max)``
            for the affine crop.  Defaults to (0.5, 1.5).
        max_translate_ratio (float): Max translation as ratio of
            ``img_scale``.  Defaults to 0.1.
        max_shear_degree (float): Max shear for affine crop (degrees).
            Defaults to 2.0.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float | tuple[float, float, float] = 114.0,
        p: float = 1.0,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_rotate_degree: float = 10.0,
        scaling_ratio_range: tuple[float, float] = (0.5, 1.5),
        max_translate_ratio: float = 0.1,
        max_shear_degree: float = 2.0,
    ) -> None:
        super().__init__()

        if not isinstance(img_scale, (tuple, list)):
            msg = "img_scale must be a tuple or list"
            raise TypeError(msg)
        if not 0 <= p <= 1.0:
            msg = f"probability must be in [0, 1], got {p}"
            raise ValueError(msg)
        if max_cached_images < 4:
            msg = f"max_cached_images must be >= 4, got {max_cached_images}"
            raise ValueError(msg)

        self.img_scale = tuple(img_scale)  # (H, W)
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = p
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

        # Affine crop parameters
        self.max_rotate_degree = max_rotate_degree
        self.scaling_ratio_range = tuple(scaling_ratio_range)
        self.max_translate_ratio = max_translate_ratio
        self.max_shear_degree = max_shear_degree

        self.results_cache: list[OTXSample] = []

    # ── Tile helpers ────────────────────────────────────────────────────

    def _resize_keep_ratio(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize image keeping aspect ratio (FILL mode)."""
        _, h, w = img.shape
        scale = max(target_h / h, target_w / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        return F.resize(img, size=[new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)

    def _resize_masks_keep_ratio(
        self, masks: torch.Tensor, target_h: int, target_w: int, orig_h: int, orig_w: int,
    ) -> torch.Tensor:
        """Resize masks keeping aspect ratio (FILL mode)."""
        scale = max(target_h / orig_h, target_w / orig_w)
        new_h = round(orig_h * scale)
        new_w = round(orig_w * scale)
        return F.resize(masks, size=[new_h, new_w], interpolation=F.InterpolationMode.NEAREST, antialias=False)

    def _compute_mosaic_params(
        self, loc: str, center_x: int, center_y: int, img_h: int, img_w: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int]:
        """Compute paste and crop coordinates for mosaic placement."""
        mosaic_h = self.img_scale[0] * 2
        mosaic_w = self.img_scale[1] * 2

        if loc == "top_left":
            x1_p = max(center_x - img_w, 0)
            y1_p = max(center_y - img_h, 0)
            x2_p = center_x
            y2_p = center_y
            x1_c = img_w - (x2_p - x1_p)
            y1_c = img_h - (y2_p - y1_p)
            x2_c = img_w
            y2_c = img_h
        elif loc == "top_right":
            x1_p = center_x
            y1_p = max(center_y - img_h, 0)
            x2_p = min(center_x + img_w, mosaic_w)
            y2_p = center_y
            x1_c = 0
            y1_c = img_h - (y2_p - y1_p)
            x2_c = min(img_w, x2_p - x1_p)
            y2_c = img_h
        elif loc == "bottom_left":
            x1_p = max(center_x - img_w, 0)
            y1_p = center_y
            x2_p = center_x
            y2_p = min(center_y + img_h, mosaic_h)
            x1_c = img_w - (x2_p - x1_p)
            y1_c = 0
            x2_c = img_w
            y2_c = min(img_h, y2_p - y1_p)
        else:  # bottom_right
            x1_p = center_x
            y1_p = center_y
            x2_p = min(center_x + img_w, mosaic_w)
            y2_p = min(center_y + img_h, mosaic_h)
            x1_c = 0
            y1_c = 0
            x2_c = min(img_w, x2_p - x1_p)
            y2_c = min(img_h, y2_p - y1_p)

        paste_coord = (x1_p, y1_p, x2_p, y2_p)
        crop_coord = (x1_c, y1_c, x2_c, y2_c)
        return paste_coord, crop_coord, x1_p - x1_c, y1_p - y1_c

    def _scale_bboxes(self, bboxes: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale bboxes by a factor."""
        return bboxes * scale

    def _translate_bboxes(self, bboxes: torch.Tensor, offset_x: int, offset_y: int) -> torch.Tensor:
        """Translate bboxes by offset."""
        if bboxes.numel() == 0:
            return bboxes
        return bboxes + bboxes.new_tensor([offset_x, offset_y, offset_x, offset_y])

    def _clip_bboxes(self, bboxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Clip bboxes to image boundary."""
        if bboxes.numel() == 0:
            return bboxes
        bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, w)
        bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, h)
        return bboxes

    def _filter_valid_bboxes(self, bboxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Get boolean mask for valid bboxes (positive area, inside image)."""
        if bboxes.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=bboxes.device)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        valid = (x2 > x1) & (y2 > y1)
        if not self.bbox_clip_border:
            valid = valid & (x2 > 0) & (y2 > 0) & (x1 < w) & (y1 < h)
        return valid

    def _create_mosaic_canvas(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Create empty 2× mosaic canvas filled with pad_val."""
        mosaic_h, mosaic_w = self.img_scale[0] * 2, self.img_scale[1] * 2
        fill_val = self.pad_val if isinstance(self.pad_val, (int, float)) else self.pad_val[0]
        if fill_val > 1.0 + 1e-5:
            fill_val = fill_val / 255.0
        return torch.full((3, mosaic_h, mosaic_w), fill_val, dtype=dtype, device=device)

    def _create_mask_canvas(self, n_masks: int, device: torch.device) -> torch.Tensor:
        """Create empty 2× mask canvas."""
        return torch.zeros((n_masks, self.img_scale[0] * 2, self.img_scale[1] * 2), dtype=torch.uint8, device=device)

    def get_indexes(self, cache: list) -> list:
        """Get 3 random indexes from cache."""
        return [int(torch.randint(0, len(cache), (1,)).item()) for _ in range(3)]

    # ── Affine crop helpers (2× canvas → 1× output) ────────────────────

    def _random_affine_matrix(self, in_h: int, in_w: int, out_h: int, out_w: int) -> torch.Tensor:
        """Build a random 3×3 affine matrix centered on the input canvas.

        The matrix maps **input** pixel coordinates to **output** pixel
        coordinates.  All geometric transforms (scale, rotation, shear) are
        applied around the *center* of the input canvas so that the mosaic
        intersection (center of the 2× canvas) maps to the center of the
        output.

        Matrix composition order:
            M = T_to_out @ Sh @ R @ S @ T_to_origin

        where T_to_origin moves the input center to the origin and T_to_out
        moves the origin to the output center (plus random translation).
        """
        angle = torch.empty(1).uniform_(-self.max_rotate_degree, self.max_rotate_degree).item()
        scale = torch.empty(1).uniform_(*self.scaling_ratio_range).item()
        shear_x = torch.empty(1).uniform_(-self.max_shear_degree, self.max_shear_degree).item()
        shear_y = torch.empty(1).uniform_(-self.max_shear_degree, self.max_shear_degree).item()
        tx = torch.empty(1).uniform_(-self.max_translate_ratio, self.max_translate_ratio).item() * out_w
        ty = torch.empty(1).uniform_(-self.max_translate_ratio, self.max_translate_ratio).item() * out_h

        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        in_cx, in_cy = in_w / 2.0, in_h / 2.0
        out_cx, out_cy = out_w / 2.0, out_h / 2.0

        # fmt: off
        # 1. Move input center to origin
        t_to_origin = torch.tensor(
            [[1, 0, -in_cx], [0, 1, -in_cy], [0, 0, 1]], dtype=torch.float32,
        )
        # 2–4. Scale, rotate, shear (all around origin)
        s_mat = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=torch.float32)
        r_mat = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32)
        sh_mat = torch.tensor([
            [1, math.tan(math.radians(shear_x)), 0],
            [math.tan(math.radians(shear_y)), 1, 0],
            [0, 0, 1],
        ], dtype=torch.float32)
        # 5. Move origin to output center + random translation
        t_to_out = torch.tensor(
            [[1, 0, out_cx + tx], [0, 1, out_cy + ty], [0, 0, 1]], dtype=torch.float32,
        )
        # fmt: on
        return t_to_out @ sh_mat @ r_mat @ s_mat @ t_to_origin

    def _warp_image(
        self, img: torch.Tensor, m_inv: torch.Tensor,
        in_h: int, in_w: int, out_h: int, out_w: int,
    ) -> torch.Tensor:
        """Warp CHW float image via grid_sample using inverse affine matrix."""
        device = img.device
        ys = torch.arange(out_h, dtype=torch.float32, device=device)
        xs = torch.arange(out_w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords_out = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
        coords_in = coords_out @ m_inv.to(device).T
        x_in, y_in = coords_in[..., 0], coords_in[..., 1]

        x_norm = 2.0 * x_in / max(in_w - 1, 1) - 1.0
        y_norm = 2.0 * y_in / max(in_h - 1, 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        warped = torch.nn.functional.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0)

        fill = self.pad_val if isinstance(self.pad_val, (int, float)) else self.pad_val[0]
        if fill > 1.0 + 1e-5:
            fill = fill / 255.0
        oob = (x_in < 0) | (x_in >= in_w) | (y_in < 0) | (y_in >= in_h)
        warped[:, oob] = fill
        return warped

    def _project_bboxes(
        self, bboxes: torch.Tensor, m_fwd: torch.Tensor, out_h: int, out_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project XYXY bboxes through forward affine matrix. Returns (bboxes, valid_mask)."""
        if bboxes.numel() == 0:
            return bboxes, torch.zeros(0, dtype=torch.bool, device=bboxes.device)

        b = bboxes.float()
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ones = torch.ones_like(x1)
        corners = torch.stack([
            torch.stack([x1, y1, ones], dim=-1),
            torch.stack([x2, y1, ones], dim=-1),
            torch.stack([x2, y2, ones], dim=-1),
            torch.stack([x1, y2, ones], dim=-1),
        ], dim=1)  # (N, 4, 3)

        projected = corners @ m_fwd.to(bboxes.device).T
        px, py = projected[..., 0], projected[..., 1]
        new_bboxes = torch.stack(
            [px.min(dim=1).values, py.min(dim=1).values,
             px.max(dim=1).values, py.max(dim=1).values], dim=-1,
        )
        if self.bbox_clip_border:
            new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clamp(0, out_w)
            new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clamp(0, out_h)

        w = new_bboxes[:, 2] - new_bboxes[:, 0]
        h = new_bboxes[:, 3] - new_bboxes[:, 1]
        valid = (w > 1) & (h > 1)
        return new_bboxes, valid

    def _warp_masks(
        self, masks: torch.Tensor, m_inv: torch.Tensor,
        in_h: int, in_w: int, out_h: int, out_w: int,
    ) -> torch.Tensor:
        """Warp NxHxW masks via grid_sample with nearest interpolation."""
        if masks.numel() == 0:
            return torch.zeros((0, out_h, out_w), dtype=masks.dtype, device=masks.device)
        device = masks.device
        ys = torch.arange(out_h, dtype=torch.float32, device=device)
        xs = torch.arange(out_w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords_out = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
        coords_in = coords_out @ m_inv.to(device).T
        x_norm = 2.0 * coords_in[..., 0] / max(in_w - 1, 1) - 1.0
        y_norm = 2.0 * coords_in[..., 1] / max(in_h - 1, 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
        return (
            torch.nn.functional.grid_sample(
                masks.unsqueeze(1).float(), grid.expand(masks.shape[0], -1, -1, -1),
                mode="nearest", padding_mode="zeros", align_corners=True,
            ).squeeze(1).to(masks.dtype)
        )

    def _apply_affine_crop(self, inputs: OTXSample, in_h: int, in_w: int) -> OTXSample:
        """Apply random affine warp from (in_h, in_w) → img_scale output."""
        out_h, out_w = self.img_scale
        m_fwd = self._random_affine_matrix(in_h, in_w, out_h, out_w)
        m_inv = torch.linalg.inv(m_fwd)

        inputs.image = self._warp_image(inputs.image, m_inv, in_h, in_w, out_h, out_w).clamp(0, 1)

        if hasattr(inputs, "bboxes") and inputs.bboxes is not None and len(inputs.bboxes) > 0:
            new_bboxes, valid = self._project_bboxes(inputs.bboxes, m_fwd, out_h, out_w)
            inputs.bboxes = tv_tensors.BoundingBoxes(new_bboxes[valid], format="XYXY", canvas_size=(out_h, out_w))
            inputs.label = inputs.label[valid]

            if hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0:
                warped_masks = self._warp_masks(inputs.masks, m_inv, in_h, in_w, out_h, out_w)
                inputs.masks = tv_tensors.Mask(warped_masks[valid])
        elif hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0:
            inputs.masks = tv_tensors.Mask(
                self._warp_masks(inputs.masks, m_inv, in_h, in_w, out_h, out_w),
            )

        inputs.img_info = _resized_crop_image_info(inputs.img_info, (out_h, out_w))
        return inputs

    def _resize_to_target(self, inputs: OTXSample) -> OTXSample:
        """Simple resize to img_scale (used when mosaic is skipped)."""
        _, in_h, in_w = inputs.image.shape
        out_h, out_w = self.img_scale
        if in_h == out_h and in_w == out_w:
            return inputs

        inputs.image = F.resize(
            inputs.image, [out_h, out_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True,
        ).clamp(0, 1)

        if hasattr(inputs, "bboxes") and inputs.bboxes is not None and len(inputs.bboxes) > 0:
            sx, sy = out_w / max(in_w, 1), out_h / max(in_h, 1)
            b = inputs.bboxes.float()
            b[:, 0::2] *= sx
            b[:, 1::2] *= sy
            inputs.bboxes = tv_tensors.BoundingBoxes(b, format="XYXY", canvas_size=(out_h, out_w))

        if hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0:
            inputs.masks = tv_tensors.Mask(
                F.resize(inputs.masks, [out_h, out_w], interpolation=F.InterpolationMode.NEAREST, antialias=False),
            )

        inputs.img_info = _resized_crop_image_info(inputs.img_info, (out_h, out_w))
        return inputs

    # ── Forward ─────────────────────────────────────────────────────────

    @typing.no_type_check
    def forward(self, *_inputs: OTXSample) -> OTXSample:
        """Apply CachedMosaic + affine crop augmentation.

        Output is always ``img_scale`` regardless of whether mosaic fires.
        """
        assert len(_inputs) == 1, "Only single sample input is supported"  # noqa: S101
        inputs = _inputs[0]

        # Add to cache
        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = int(torch.randint(0, len(self.results_cache), (1,)).item()) if self.random_pop else 0
            self.results_cache.pop(index)

        # Return early if cache too small — resize to target
        if len(self.results_cache) < 4:
            return self._resize_to_target(inputs)

        # Skip with probability — resize to target
        if torch.rand(1).item() > self.prob:
            return self._resize_to_target(inputs)

        # ── Build 2× mosaic canvas ──────────────────────────────────────
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        target_h, target_w = self.img_scale
        mosaic_h, mosaic_w = target_h * 2, target_w * 2

        center_x = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_w)
        center_y = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_h)

        img_tensor = inputs.image
        device = img_tensor.device
        mosaic_img = self._create_mosaic_canvas(img_tensor.dtype, device)

        all_bboxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        samples = [inputs, *mix_results]

        for i, loc in enumerate(loc_strs):
            sample = samples[i]
            img_i = sample.image
            _, orig_h, orig_w = img_i.shape

            scale = max(target_h / orig_h, target_w / orig_w)
            img_i = self._resize_keep_ratio(img_i, target_h, target_w)
            _, new_h, new_w = img_i.shape

            paste_coord, crop_coord, pad_w, pad_h = self._compute_mosaic_params(
                loc, center_x, center_y, new_h, new_w,
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            mosaic_img[:, y1_p:y2_p, x1_p:x2_p] = img_i[:, y1_c:y2_c, x1_c:x2_c]

            bboxes_i = self._translate_bboxes(self._scale_bboxes(sample.bboxes.float(), scale), pad_w, pad_h)
            all_bboxes.append(bboxes_i)
            all_labels.append(sample.label)

            if with_mask:
                masks_i = sample.masks
                if masks_i is not None and len(masks_i) > 0:
                    masks_i = self._resize_masks_keep_ratio(masks_i, target_h, target_w, orig_h, orig_w)
                    n_masks = masks_i.shape[0]
                    mask_canvas = self._create_mask_canvas(n_masks, device)
                    mask_canvas[:, y1_p:y2_p, x1_p:x2_p] = masks_i[:, y1_c:y2_c, x1_c:x2_c]
                    all_masks.append(mask_canvas)

        mosaic_bboxes = torch.cat(all_bboxes, dim=0)
        mosaic_labels = torch.cat(all_labels, dim=0)

        if self.bbox_clip_border:
            mosaic_bboxes = self._clip_bboxes(mosaic_bboxes, mosaic_h, mosaic_w)

        valid_mask = self._filter_valid_bboxes(mosaic_bboxes, mosaic_h, mosaic_w)
        mosaic_bboxes = mosaic_bboxes[valid_mask]
        mosaic_labels = mosaic_labels[valid_mask]

        inputs.image = mosaic_img.clamp(0, 1)
        inputs.bboxes = tv_tensors.BoundingBoxes(mosaic_bboxes, format="XYXY", canvas_size=(mosaic_h, mosaic_w))
        inputs.label = mosaic_labels

        if with_mask and len(all_masks) > 0:
            mosaic_masks = torch.cat(all_masks, dim=0)[valid_mask]
            inputs.masks = tv_tensors.Mask(mosaic_masks)

        # ── Affine crop: 2× canvas → 1× target ─────────────────────────
        return self._apply_affine_crop(inputs, mosaic_h, mosaic_w)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"img_scale={self.img_scale}, "
            f"center_ratio_range={self.center_ratio_range}, "
            f"pad_val={self.pad_val}, "
            f"prob={self.prob}, "
            f"max_cached_images={self.max_cached_images}, "
            f"random_pop={self.random_pop}, "
            f"scaling_ratio_range={self.scaling_ratio_range}, "
            f"max_rotate_degree={self.max_rotate_degree})"
        )


class CachedMixUp(tvt_v2.Transform):
    """Geometry-preserved MixUp augmentation for detection / instance segmentation.

    Blends the current image with a cached image using alpha blending without
    any geometric transformation.  Bounding boxes, labels and masks from both images are
    concatenated.

    If the cached image has a different spatial size it is resized to the current
    image size with ``keep_aspect_ratio`` padding so the aspect ratio is preserved.

    Args:
        img_scale (tuple[int, int]): Reference image size ``(H, W)`` used when
            the cached image needs to be rescaled.  Defaults to (640, 640).
        mix_ratio (float): Blending ratio ``β``.  The mixed image is
            ``β * current + (1-β) * cached``.  0.5 = equal blend.
            Defaults to 0.5.
        pad_val (float): Padding fill value ([0, 255] or [0, 1]).
            Defaults to 114.0.
        max_iters (int): Max attempts to find a non-empty cached sample.
            Defaults to 15.
        bbox_clip_border (bool): Clip bboxes to image boundary after concat.
            Defaults to True.
        max_cached_images (int): Maximum cache size (>= 2). Defaults to 20.
        random_pop (bool): Random vs FIFO eviction. Defaults to True.
        p (float): Probability of applying MixUp. Defaults to 1.0.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        mix_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__()

        if not isinstance(img_scale, (tuple, list)):
            msg = "img_scale must be a tuple or list"
            raise TypeError(msg)
        if max_cached_images < 2:
            msg = f"Cache size must be >= 2, got {max_cached_images}"
            raise ValueError(msg)
        if not 0 <= p <= 1.0:
            msg = f"Probability must be in [0,1], got {p}"
            raise ValueError(msg)

        self.img_scale = tuple(img_scale)  # (H, W)
        self.mix_ratio = mix_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = p

        self.results_cache: list[OTXSample] = []

    def _get_cached_index(self) -> int:
        """Return index of a cached sample with non-empty bboxes."""
        index = 0
        for _ in range(self.max_iters):
            index = int(torch.randint(0, len(self.results_cache), (1,)).item())
            if len(getattr(self.results_cache[index], "bboxes", [])) > 0:
                return index
        return index

    def _match_size(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize *img* to ``(target_h, target_w)`` with keep-ratio + pad."""
        _, h, w = img.shape
        if h == target_h and w == target_w:
            return img
        scale = min(target_h / h, target_w / w)
        new_h, new_w = round(h * scale), round(w * scale)
        resized = F.resize(img, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        pad_val = self.pad_val / 255.0 if self.pad_val > 1.0 + 1e-5 else self.pad_val
        canvas = torch.full((img.shape[0], target_h, target_w), pad_val, dtype=img.dtype, device=img.device)
        canvas[:, :new_h, :new_w] = resized
        return canvas

    def _match_masks_size(
        self, masks: torch.Tensor, src_h: int, src_w: int, target_h: int, target_w: int,
    ) -> torch.Tensor:
        """Resize masks to ``(target_h, target_w)`` with keep-ratio + zero-pad."""
        if masks.numel() == 0:
            return torch.zeros((0, target_h, target_w), dtype=masks.dtype, device=masks.device)
        if src_h == target_h and src_w == target_w:
            return masks
        scale = min(target_h / src_h, target_w / src_w)
        new_h, new_w = round(src_h * scale), round(src_w * scale)
        resized = F.resize(masks, [new_h, new_w], interpolation=F.InterpolationMode.NEAREST, antialias=False)
        canvas = torch.zeros((masks.shape[0], target_h, target_w), dtype=masks.dtype, device=masks.device)
        canvas[:, :new_h, :new_w] = resized
        return canvas

    def _scale_bboxes(self, bboxes: torch.Tensor, src_h: int, src_w: int, dst_h: int, dst_w: int) -> torch.Tensor:
        """Scale XYXY bboxes from (src_h, src_w) to (dst_h, dst_w) keeping ratio + pad."""
        if bboxes.numel() == 0:
            return bboxes
        scale = min(dst_h / src_h, dst_w / src_w)
        return bboxes * scale


    @typing.no_type_check
    def forward(self, *_inputs: OTXSample) -> OTXSample:
        """Apply MixUp transform using pure torch operations."""
        assert len(_inputs) == 1, "Multiple inputs not supported"  # noqa: S101
        inputs = _inputs[0]

        # Cache management
        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            pop_idx = int(torch.randint(0, len(self.results_cache), (1,)).item()) if self.random_pop else 0
            self.results_cache.pop(pop_idx)

        # Early returns
        if len(self.results_cache) <= 1:
            return inputs

        if torch.rand(1).item() > self.prob:
            return inputs

        # Get cached sample
        cache_idx = self._get_cached_index()
        cached = copy.deepcopy(self.results_cache[cache_idx])

        if cached.bboxes.shape[0] == 0:
            return inputs

        # ---- sizes ----
        ori_img = inputs.image                       # (C, H, W)  float [0,1]
        _, target_h, target_w = ori_img.shape
        cached_img = cached.image
        _, cached_h, cached_w = cached_img.shape

        # ---- blend ratio ----
        beta = self.mix_ratio

        # ---- match cached image to current image size (geometry-preserved) ----
        cached_img = self._match_size(cached_img, target_h, target_w)

        # ---- alpha blend ----
        mixup_img = (beta * ori_img + (1.0 - beta) * cached_img).clamp(0, 1)

        # ---- rescale cached bboxes to the current canvas ----
        cached_bboxes = cached.bboxes.float()
        cached_bboxes = self._scale_bboxes(cached_bboxes, cached_h, cached_w, target_h, target_w)
        if self.bbox_clip_border and cached_bboxes.numel() > 0:
            cached_bboxes[..., 0::2] = cached_bboxes[..., 0::2].clamp(0, target_w)
            cached_bboxes[..., 1::2] = cached_bboxes[..., 1::2].clamp(0, target_h)

        # ---- concatenate annotations ----
        mixup_bboxes = torch.cat([inputs.bboxes.float(), cached_bboxes], dim=0)
        mixup_labels = torch.cat([inputs.label, cached.label], dim=0)

        # ---- filter degenerate boxes ----
        if mixup_bboxes.numel() > 0:
            w = mixup_bboxes[:, 2] - mixup_bboxes[:, 0]
            h = mixup_bboxes[:, 3] - mixup_bboxes[:, 1]
            valid = (w > 0) & (h > 0)
            mixup_bboxes = mixup_bboxes[valid]
            mixup_labels = mixup_labels[valid]
        else:
            valid = torch.zeros(0, dtype=torch.bool)

        # ---- masks (instance segmentation) ----
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None
        if with_mask:
            ori_masks = inputs.masks
            cached_masks = getattr(cached, "masks", None)
            if cached_masks is not None and len(cached_masks) > 0:
                cached_masks = self._match_masks_size(cached_masks, cached_h, cached_w, target_h, target_w)
                all_masks = torch.cat([ori_masks, cached_masks], dim=0)
                all_masks = all_masks[valid] if valid.numel() > 0 else all_masks
            else:
                n_cached_labels = len(cached.label)
                all_valid = valid[:len(ori_masks)] if valid.numel() > 0 else torch.ones(len(ori_masks), dtype=torch.bool)
                all_masks = ori_masks[all_valid]
            inputs.masks = tv_tensors.Mask(all_masks)

        # ---- write back ----
        inputs.image = mixup_img
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_bboxes, format="XYXY", canvas_size=(target_h, target_w))
        inputs.label = mixup_labels
        inputs.img_info = _resized_crop_image_info(inputs.img_info, (target_h, target_w))

        return inputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"img_scale={self.img_scale}, "
            f"mix_ratio={self.mix_ratio}, "
            f"pad_val={self.pad_val}, "
            f"max_cached_images={self.max_cached_images}, "
            f"random_pop={self.random_pop}, "
            f"prob={self.prob})"
        )


class RandomIoUCrop(tvt_v2.RandomIoUCrop):
    """Random IoU crop with the option to set probability.

    Args:
        min_scale (float, optional): the same as RandomIoUCrop. Defaults to 0.3.
        max_scale (float, optional): the same as RandomIoUCrop. Defaults to 1.
        min_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 0.5.
        max_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 2.
        sampler_options (list[float] | None, optional): the same as RandomIoUCrop. Defaults to None.
        trials (int, optional): the same as RandomIoUCrop. Defaults to 40.
        p (float, optional): probability of applying the crop. Defaults to 1.0.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:  # noqa: ANN401
        """Apply the transform to the given inputs."""
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().__call__(*inputs)
