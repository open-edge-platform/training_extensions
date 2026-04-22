# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter: VisionDataset sample -> Ultralytics sample dict."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from getitune.data.entity.base import ImageInfo

from .geometry import build_ratio_pad, xyxy_abs_to_xywh_norm

logger = logging.getLogger(__name__)


class UltralyticsDatasetAdapter(TorchDataset):
    """Wraps a getitune ``VisionDataset`` and yields Ultralytics sample dicts.

    Each call to ``__getitem__`` pulls an already-augmented sample
    (float32 CHW ``[0, 1]``) from the underlying ``VisionDataset`` and
    converts it to the dict format that Ultralytics trainers/validators
    expect.

    The adapter does **not** apply any additional augmentation.
    """

    def __init__(self, vision_dataset: TorchDataset, *, include_masks: bool = False) -> None:
        self._dataset = vision_dataset
        self._include_masks = include_masks

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Convert a single getitune sample to an Ultralytics sample dict."""
        sample = self._dataset[index]

        img: torch.Tensor = sample.image  # float32 CHW [0, 1]
        if img.ndim != 3:
            msg = f"Expected 3-D image tensor (CHW), got shape {img.shape}"
            raise ValueError(msg)

        _, h, w = img.shape

        # --- Geometry from ImageInfo ---
        img_info: ImageInfo | None = getattr(sample, "img_info", None)
        if img_info is not None:
            ori_shape = img_info.ori_shape
            padding = img_info.padding
        else:
            ori_shape = (h, w)
            padding = (0, 0, 0, 0)

        ratio_pad = build_ratio_pad(ori_shape, (h, w), padding)

        # --- Bounding boxes ---
        bboxes_raw = getattr(sample, "bboxes", None)
        if bboxes_raw is not None and len(bboxes_raw) > 0:
            bboxes_xywh = xyxy_abs_to_xywh_norm(bboxes_raw, img_w=w, img_h=h)
        else:
            bboxes_xywh = np.zeros((0, 4), dtype=np.float32)

        # --- Class labels ---
        labels_raw = getattr(sample, "label", None)
        if labels_raw is not None:
            if isinstance(labels_raw, torch.Tensor):
                cls = labels_raw.detach().cpu().numpy().astype(np.float32)
            else:
                cls = np.asarray(labels_raw, dtype=np.float32)
            if cls.ndim == 1:
                cls = cls[:, np.newaxis]  # (N,) -> (N, 1)
        else:
            cls = np.zeros((0, 1), dtype=np.float32)

        result: dict[str, Any] = {
            "im_file": "",
            "img": img,
            "cls": cls,
            "bboxes": bboxes_xywh,
            "batch_idx": torch.zeros(cls.shape[0], dtype=torch.float32),
            "ori_shape": ori_shape,
            "resized_shape": (h, w),
            "ratio_pad": ratio_pad,
        }

        # --- Instance masks (for segmentation) ---
        if self._include_masks:
            masks_raw = getattr(sample, "masks", None)
            if masks_raw is not None and len(masks_raw) > 0:
                mask_tensor = masks_raw
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np = mask_tensor.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(mask_tensor)
                # Ultralytics expects masks as (N, H, W) uint8.
                if mask_np.ndim == 2:
                    mask_np = mask_np[np.newaxis]
                result["masks"] = torch.from_numpy(mask_np.astype(np.float32))
            else:
                result["masks"] = torch.zeros((0, h, w), dtype=torch.float32)

        return result
