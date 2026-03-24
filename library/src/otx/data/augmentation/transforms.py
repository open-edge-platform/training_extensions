# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom image transforms for OTX augmentation pipeline."""

from __future__ import annotations

import copy
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
    """Mosaic augmentation with caching using pure torchvision operations.

    Combines four images into a single mosaic image by placing them in quadrants
    around a randomly chosen center point. Uses caching to improve randomness
    without requiring dataset access.

    This implementation uses only torch/torchvision operations, no numpy/cv2.

    Args:
        img_scale (tuple[int, int]): Target image size (height, width) for each
            image before creating mosaic. Defaults to (640, 640).
        center_ratio_range (tuple[float, float]): Range for random center position
            as ratio of img_scale. Defaults to (0.5, 1.5).
        bbox_clip_border (bool): Whether to clip bboxes to image boundary.
            Defaults to True.
        pad_val (float | tuple[float, float, float]): Padding value for mosaic canvas.
            Defaults to 114.0.
        p (float): Probability of applying mosaic. Defaults to 1.0.
        max_cached_images (int): Maximum number of cached images. Defaults to 40.
        random_pop (bool): If True, randomly remove cached images when full.
            If False, use FIFO. Defaults to True.
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

        self.results_cache: list[OTXSample] = []

    def _resize_keep_ratio(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize image keeping aspect ratio using torchvision.

        Args:
            img: CHW tensor image.
            target_h: Target height.
            target_w: Target width.

        Returns:
            Resized CHW tensor that fits within target size.
        """
        _, h, w = img.shape
        scale = min(target_h / h, target_w / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        return F.resize(img, size=[new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)

    def _resize_masks_keep_ratio(
        self,
        masks: torch.Tensor,
        target_h: int,
        target_w: int,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Resize masks keeping aspect ratio using torchvision.

        Args:
            masks: NxHxW tensor masks.
            target_h: Target height.
            target_w: Target width.
            orig_h: Original image height (for scale calculation).
            orig_w: Original image width (for scale calculation).

        Returns:
            Resized masks that match the resized image dimensions.
        """
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = round(orig_h * scale)
        new_w = round(orig_w * scale)
        return F.resize(masks, size=[new_h, new_w], interpolation=F.InterpolationMode.NEAREST, antialias=False)

    def _compute_mosaic_params(
        self,
        loc: str,
        center_x: int,
        center_y: int,
        img_h: int,
        img_w: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int]:
        """Compute paste and crop coordinates for mosaic placement.

        Args:
            loc: Position string ("top_left", "top_right", "bottom_left", "bottom_right").
            center_x: X coordinate of mosaic center.
            center_y: Y coordinate of mosaic center.
            img_h: Height of the image to place.
            img_w: Width of the image to place.

        Returns:
            paste_coord: (x1, y1, x2, y2) coordinates in mosaic canvas.
            crop_coord: (x1, y1, x2, y2) coordinates in source image.
            pad_w: Horizontal offset for bbox/mask adjustment.
            pad_h: Vertical offset for bbox/mask adjustment.
        """
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
        pad_w = x1_p - x1_c
        pad_h = y1_p - y1_c

        return paste_coord, crop_coord, pad_w, pad_h

    def _scale_bboxes(self, bboxes: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale bboxes by a factor (pure torch)."""
        return bboxes * scale

    def _translate_bboxes(self, bboxes: torch.Tensor, offset_x: int, offset_y: int) -> torch.Tensor:
        """Translate bboxes by offset (pure torch)."""
        if bboxes.numel() == 0:
            return bboxes
        offset = bboxes.new_tensor([offset_x, offset_y, offset_x, offset_y])
        return bboxes + offset

    def _clip_bboxes(self, bboxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Clip bboxes to image boundary (pure torch)."""
        if bboxes.numel() == 0:
            return bboxes
        bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, w)
        bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, h)
        return bboxes

    def _filter_valid_bboxes(self, bboxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Get mask for valid bboxes."""
        if bboxes.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=bboxes.device)

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        valid = (x2 > x1) & (y2 > y1)

        # If clipping is disabled, still ensure boxes overlap image at least partially.
        if not self.bbox_clip_border:
            inside = (x2 > 0) & (y2 > 0) & (x1 < w) & (y1 < h)
            valid = valid & inside

        return valid

    def _create_mosaic_canvas(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Create empty mosaic canvas filled with pad_val."""
        mosaic_h = self.img_scale[0] * 2
        mosaic_w = self.img_scale[1] * 2

        fill_val = self.pad_val if isinstance(self.pad_val, (int, float)) else self.pad_val[0]
        if fill_val > 1.0 + 1e-5:
            fill_val = fill_val / 255.0
        return torch.full((3, mosaic_h, mosaic_w), fill_val, dtype=dtype, device=device)

    def _create_mask_canvas(self, n_masks: int, device: torch.device) -> torch.Tensor:
        """Create empty mask canvas."""
        mosaic_h = self.img_scale[0] * 2
        mosaic_w = self.img_scale[1] * 2
        return torch.zeros((n_masks, mosaic_h, mosaic_w), dtype=torch.uint8, device=device)

    def get_indexes(self, cache: list) -> list:
        """Get random indexes from cache."""
        return [int(torch.randint(0, len(cache), (1,)).item()) for _ in range(3)]

    @typing.no_type_check
    def forward(self, *_inputs: OTXSample) -> OTXSample:
        """Apply CachedMosaic augmentation.

        Args:
            _inputs: Single OTXSample input.

        Returns:
            Augmented OTXSample with mosaic image.
        """
        assert len(_inputs) == 1, "Only single sample input is supported"  # noqa: S101
        inputs = _inputs[0]

        # Add to cache
        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = int(torch.randint(0, len(self.results_cache), (1,)).item()) if self.random_pop else 0
            self.results_cache.pop(index)

        # Return early if cache too small
        if len(self.results_cache) < 4:
            return inputs

        # Skip with probability
        if torch.rand(1).item() > self.prob:
            return inputs

        # Get 3 additional samples from cache
        indices = self.get_indexes(self.results_cache)
        mix_results = [self.results_cache[i] for i in indices]

        # Prepare mosaic
        target_h, target_w = self.img_scale
        mosaic_h, mosaic_w = target_h * 2, target_w * 2

        # Random center position
        center_x = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_w)
        center_y = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_h)

        # Convert input image to tensor
        img_tensor = inputs.image
        device = img_tensor.device

        # Create mosaic canvas
        mosaic_img = self._create_mosaic_canvas(img_tensor.dtype, device)

        # Collect all bboxes, labels, masks
        all_bboxes = []
        all_labels = []
        all_masks = []
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        samples = [inputs, *mix_results]
        # random tile order assignment.
        order = torch.randperm(len(samples)).tolist()
        samples = [samples[idx] for idx in order]

        for i, loc in enumerate(loc_strs):
            sample = samples[i]

            # Convert image to tensor
            img_i = sample.image
            _, orig_h, orig_w = img_i.shape

            # Resize keeping aspect ratio
            scale = min(target_h / orig_h, target_w / orig_w)
            img_i = self._resize_keep_ratio(img_i, target_h, target_w)
            _, new_h, new_w = img_i.shape

            # Compute paste/crop coordinates
            paste_coord, crop_coord, pad_w, pad_h = self._compute_mosaic_params(
                loc,
                center_x,
                center_y,
                new_h,
                new_w,
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # Paste image region
            mosaic_img[:, y1_p:y2_p, x1_p:x2_p] = img_i[:, y1_c:y2_c, x1_c:x2_c]

            # Transform bboxes
            bboxes_i = sample.bboxes.float()

            bboxes_i = self._scale_bboxes(bboxes_i, scale)
            bboxes_i = self._translate_bboxes(bboxes_i, pad_w, pad_h)
            all_bboxes.append(bboxes_i)

            # Collect labels
            labels_i = sample.label
            all_labels.append(labels_i)

            # Transform masks if present
            if with_mask:
                masks_i = sample.masks
                if masks_i is not None and len(masks_i) > 0:
                    # Resize masks
                    masks_i = self._resize_masks_keep_ratio(masks_i, target_h, target_w, orig_h, orig_w)

                    # Create canvas for this sample's masks and paste
                    n_masks = masks_i.shape[0]
                    mask_canvas = self._create_mask_canvas(n_masks, device)
                    mask_canvas[:, y1_p:y2_p, x1_p:x2_p] = masks_i[:, y1_c:y2_c, x1_c:x2_c]
                    all_masks.append(mask_canvas)

        # Concatenate all bboxes and labels
        mosaic_bboxes = torch.cat(all_bboxes, dim=0)
        mosaic_labels = torch.cat(all_labels, dim=0)

        # Clip bboxes if needed
        if self.bbox_clip_border:
            mosaic_bboxes = self._clip_bboxes(mosaic_bboxes, mosaic_h, mosaic_w)

        # Filter valid bboxes
        valid_mask = self._filter_valid_bboxes(mosaic_bboxes, mosaic_h, mosaic_w)
        mosaic_bboxes = mosaic_bboxes[valid_mask]
        mosaic_labels = mosaic_labels[valid_mask]

        # Update inputs
        inputs.image = mosaic_img.clamp(0, 1)
        inputs.img_info = _resized_crop_image_info(inputs.img_info, (mosaic_h, mosaic_w))
        inputs.bboxes = tv_tensors.BoundingBoxes(
            mosaic_bboxes,
            format="XYXY",
            canvas_size=(mosaic_h, mosaic_w),
        )
        inputs.label = mosaic_labels

        # Handle masks
        if with_mask and len(all_masks) > 0:
            mosaic_masks = torch.cat(all_masks, dim=0)
            mosaic_masks = mosaic_masks[valid_mask]
            inputs.masks = tv_tensors.Mask(mosaic_masks)

        return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop}, "
        return repr_str


class CachedMixUp(tvt_v2.Transform):
    """Pure-torch MixUp augmentation for object detection and instance segmentation.

    Mixes the current image with a cached image using alpha blending.
    All operations use pure torch - no numpy or cv2.

    Args:
        img_scale (Sequence[int]): Target image size (H, W). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale jitter ratio range. Defaults to (0.5, 1.5).
        flip_ratio (float): Probability of horizontal flip. Defaults to 0.5.
        pad_val (float): Padding value (0-255 or 0-1 for float). Defaults to 114.0.
        max_iters (int): Max iterations to find non-empty cached sample. Defaults to 15.
        bbox_clip_border (bool): Whether to clip bboxes to image border. Defaults to True.
        max_cached_images (int): Maximum cache size. Defaults to 20.
        random_pop (bool): Random vs FIFO cache eviction. Defaults to True.
        p (float): Probability of applying mixup. Defaults to 1.0.
        mix_ratio (float): Blending ratio (0.5 = equal mix). Defaults to 0.5.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        ratio_range: tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        p: float = 1.0,
        mix_ratio: float = 0.5,
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
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = p
        self.mix_ratio = mix_ratio

        self.results_cache: list[OTXSample] = []

    def _resize_keep_ratio(self, img: torch.Tensor, target_size: tuple[int, int]) -> tuple[torch.Tensor, float]:
        """Resize image keeping aspect ratio using torchvision.

        Args:
            img: CHW tensor
            target_size: (H, W) target size

        Returns:
            Resized image and scale ratio
        """
        _, h, w = img.shape
        target_h, target_w = target_size
        scale_ratio = min(target_h / h, target_w / w)

        new_h = int(h * scale_ratio)
        new_w = int(w * scale_ratio)

        resized = F.resize(img, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        return resized, scale_ratio

    def _scale_bboxes(self, bboxes: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale bboxes by a factor."""
        if bboxes.numel() == 0:
            return bboxes
        return bboxes * scale

    def _translate_bboxes(self, bboxes: torch.Tensor, offset_x: float, offset_y: float) -> torch.Tensor:
        """Translate bboxes by offset."""
        if bboxes.numel() == 0:
            return bboxes
        bboxes[:, 0] += offset_x
        bboxes[:, 1] += offset_y
        bboxes[:, 2] += offset_x
        bboxes[:, 3] += offset_y
        return bboxes

    def _clip_bboxes(self, bboxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """Clip bboxes to image boundaries."""
        if bboxes.numel() == 0:
            return bboxes
        bboxes[:, 0].clamp_(0, img_w)
        bboxes[:, 1].clamp_(0, img_h)
        bboxes[:, 2].clamp_(0, img_w)
        bboxes[:, 3].clamp_(0, img_h)
        return bboxes

    def _flip_bboxes_horizontal(self, bboxes: torch.Tensor, img_w: int) -> torch.Tensor:
        """Flip bboxes horizontally."""
        if bboxes.numel() == 0:
            return bboxes
        x1 = img_w - bboxes[:, 2]
        x2 = img_w - bboxes[:, 0]
        bboxes[:, 0] = x1
        bboxes[:, 2] = x2
        return bboxes

    def _filter_valid_bboxes(
        self,
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        img_h: int,
        img_w: int,
        min_area: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter bboxes that are inside image and have valid area.

        Returns:
            Filtered bboxes, labels, and boolean mask of valid indices
        """
        if bboxes.numel() == 0:
            return bboxes, labels, torch.zeros(0, dtype=torch.bool)

        # Check if bbox center is inside image
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        inside = (cx >= 0) & (cx < img_w) & (cy >= 0) & (cy < img_h)

        # Check valid area
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        valid_area = (w > 0) & (h > 0) & (w * h >= min_area)

        valid = inside & valid_area
        return bboxes[valid], labels[valid], valid

    def _get_cached_index(self) -> int:
        """Get index of cached sample with non-empty bboxes."""
        index = 0
        for _ in range(self.max_iters):
            index = int(torch.randint(0, len(self.results_cache), (1,)).item())
            if len(getattr(self.results_cache[index], "bboxes", [])) > 0:
                return index
        return index

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

        ori_img = inputs.image
        cached_img = cached.image

        _, target_h, target_w = ori_img.shape
        pad_val = self.pad_val / 255.0 if self.pad_val > 1.0 else self.pad_val

        # Check for masks
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None

        # Random parameters
        jit_factor = float(torch.empty(1).uniform_(*self.ratio_range).item())
        do_flip = torch.rand(1).item() < self.flip_ratio

        # Step 1: Resize cached image keeping aspect ratio
        cached_resized, scale_ratio = self._resize_keep_ratio(cached_img, self.img_scale)
        c, resized_h, resized_w = cached_resized.shape

        # Step 2: Paste onto padded canvas
        canvas = torch.full((c, self.img_scale[0], self.img_scale[1]), pad_val, dtype=cached_resized.dtype)
        paste_h = min(resized_h, self.img_scale[0])
        paste_w = min(resized_w, self.img_scale[1])
        canvas[:, :paste_h, :paste_w] = cached_resized[:, :paste_h, :paste_w]

        # Step 3: Apply scale jitter
        combined_scale = scale_ratio * jit_factor
        jit_h = int(self.img_scale[0] * jit_factor)
        jit_w = int(self.img_scale[1] * jit_factor)
        canvas_jittered = F.resize(canvas, [jit_h, jit_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)

        # Step 4: Horizontal flip
        if do_flip:
            canvas_jittered = canvas_jittered.flip(-1)

        # Step 5: Pad and random crop
        jit_h, jit_w = canvas_jittered.shape[-2:]
        pad_h = max(jit_h, target_h)
        pad_w = max(jit_w, target_w)

        padded = torch.full((c, pad_h, pad_w), pad_val, dtype=canvas_jittered.dtype)
        padded[:, :jit_h, :jit_w] = canvas_jittered

        # Random crop offset
        y_offset = int(torch.randint(0, max(1, pad_h - target_h + 1), (1,)).item()) if pad_h > target_h else 0
        x_offset = int(torch.randint(0, max(1, pad_w - target_w + 1), (1,)).item()) if pad_w > target_w else 0

        cropped = padded[:, y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # Step 6: Transform bboxes
        cached_bboxes = cached.bboxes.float()

        # Scale bboxes
        cached_bboxes = self._scale_bboxes(cached_bboxes, combined_scale)

        # Clip before flip
        if self.bbox_clip_border:
            cached_bboxes = self._clip_bboxes(cached_bboxes, jit_h, jit_w)

        # Flip bboxes
        if do_flip:
            cached_bboxes = self._flip_bboxes_horizontal(cached_bboxes, jit_w)

        # Translate bboxes (account for crop offset)
        cached_bboxes = self._translate_bboxes(cached_bboxes, -x_offset, -y_offset)

        # Clip after translate
        if self.bbox_clip_border:
            cached_bboxes = self._clip_bboxes(cached_bboxes, target_h, target_w)

        # Step 7: Mix images (alpha blending)
        beta = self.mix_ratio
        mixup_img = beta * ori_img + (1.0 - beta) * cropped

        # Step 8: Combine bboxes and labels
        ori_bboxes = inputs.bboxes.float()
        ori_labels = inputs.label
        cached_labels = cached.label
        # Filter valid cached bboxes
        cached_bboxes, cached_labels, valid_mask = self._filter_valid_bboxes(
            cached_bboxes, cached_labels, target_h, target_w
        )

        # Concatenate
        mixup_bboxes = torch.cat([ori_bboxes, cached_bboxes], dim=0)
        mixup_labels = torch.cat([ori_labels, cached_labels], dim=0)

        # Step 9: Handle masks for instance segmentation
        if with_mask:
            ori_masks = inputs.masks
            cached_masks = cached.masks

            # Transform cached masks - fully vectorized (one interpolate call for all instances)
            if len(cached_masks) > 0:
                n_m = cached_masks.shape[0]
                mh, mw = cached_masks.shape[-2], cached_masks.shape[-1]
                new_h = int(mh * combined_scale)
                new_w = int(mw * combined_scale)
                if new_h > 0 and new_w > 0:
                    # Batch resize: (N, H, W) -> (N, 1, H, W) -> interpolate -> (N, new_h, new_w)
                    resized_batch = (
                        torch.nn.functional.interpolate(
                            cached_masks.unsqueeze(1).float(),
                            size=(new_h, new_w),
                            mode="nearest",
                        )
                        .squeeze(1)
                        .to(torch.uint8)
                    )

                    # Batch flip
                    if do_flip:
                        resized_batch = resized_batch.flip(-1)

                    # Batch translate/crop - same src/dst region for all N masks
                    out_batch = torch.zeros((n_m, target_h, target_w), dtype=torch.uint8, device=resized_batch.device)
                    oy, ox = -y_offset, -x_offset
                    rh, rw = resized_batch.shape[-2], resized_batch.shape[-1]
                    src_y1 = max(0, oy)
                    src_y2 = min(rh, oy + target_h)
                    src_x1 = max(0, ox)
                    src_x2 = min(rw, ox + target_w)
                    dst_y1 = max(0, -oy)
                    dst_y2 = dst_y1 + (src_y2 - src_y1)
                    dst_x1 = max(0, -ox)
                    dst_x2 = dst_x1 + (src_x2 - src_x1)
                    if src_y2 > src_y1 and src_x2 > src_x1:
                        out_batch[:, dst_y1:dst_y2, dst_x1:dst_x2] = resized_batch[:, src_y1:src_y2, src_x1:src_x2]

                    # Filter by valid_mask and concatenate
                    cached_masks_transformed = out_batch[valid_mask.cpu()]
                    if len(cached_masks_transformed) > 0:
                        mixup_masks = torch.cat([ori_masks, cached_masks_transformed], dim=0)
                    else:
                        mixup_masks = ori_masks
                else:
                    mixup_masks = ori_masks
            else:
                mixup_masks = ori_masks

            inputs.masks = mixup_masks

        # Update inputs
        inputs.image = mixup_img.clamp(0, 1)
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_bboxes, format="XYXY", canvas_size=(target_h, target_w))
        inputs.label = mixup_labels
        inputs.img_info = _resized_crop_image_info(inputs.img_info, (target_h, target_w))

        return inputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"img_scale={self.img_scale}, "
            f"ratio_range={self.ratio_range}, "
            f"flip_ratio={self.flip_ratio}, "
            f"pad_val={self.pad_val}, "
            f"max_iters={self.max_iters}, "
            f"bbox_clip_border={self.bbox_clip_border}, "
            f"max_cached_images={self.max_cached_images}, "
            f"random_pop={self.random_pop}, "
            f"prob={self.prob}, "
            f"mix_ratio={self.mix_ratio})"
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
