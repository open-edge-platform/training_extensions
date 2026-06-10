# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter: VisionDataset sample -> Ultralytics sample dict."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.base import ImageInfo

from .geometry import build_ratio_pad, xyxy_abs_to_xywh_norm


class UltralyticsDatasetAdapter(TorchDataset):
    """Wrap a getitune ``VisionDataset`` as Ultralytics samples."""

    def __init__(
        self,
        vision_dataset: VisionDataset,
        *,
        include_masks: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            vision_dataset: The getitune VisionDataset to wrap.
            include_masks: Whether to include instance masks in the output.
        """
        self._dataset = vision_dataset
        self._include_masks = include_masks

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Convert a single getitune sample to an Ultralytics sample dict."""
        sample = self._dataset[index]

        img: torch.Tensor = sample.image  # float32 CHW [0, 1]
        if img.ndim != 3:
            msg = f"Expected 3-D image tensor (CHW), got shape {img.shape}"
            raise ValueError(msg)

        _, tensor_h, tensor_w = img.shape

        img_info: ImageInfo | None = getattr(sample, "img_info", None)
        if img_info is not None:
            ori_shape = img_info.ori_shape
            resized_shape = img_info.img_shape
            padding = img_info.padding
        else:
            ori_shape = (tensor_h, tensor_w)
            resized_shape = (tensor_h, tensor_w)
            padding = (0, 0, 0, 0)

        ratio_pad = build_ratio_pad(ori_shape, resized_shape, padding)

        bboxes_raw = getattr(sample, "bboxes", None)
        if bboxes_raw is not None and len(bboxes_raw) > 0:
            canvas = getattr(bboxes_raw, "canvas_size", None)
            canvas_size = cast("tuple[int, int] | None", tuple(canvas) if canvas is not None else None)
            bboxes_xywh = xyxy_abs_to_xywh_norm(
                bboxes_raw,
                img_w=tensor_w,
                img_h=tensor_h,
                canvas_size=canvas_size,
                scale_factor=img_info.scale_factor if img_info is not None else None,
                padding=padding,
            )
        else:
            bboxes_xywh = np.zeros((0, 4), dtype=np.float32)

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
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

        if self._include_masks:
            masks_raw = getattr(sample, "masks", None)
            if masks_raw is not None and len(masks_raw) > 0:
                mask_tensor = masks_raw
                if not isinstance(mask_tensor, torch.Tensor):
                    mask_tensor = torch.as_tensor(mask_tensor)
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)

                if mask_tensor.shape[1:] != (tensor_h, tensor_w):
                    pad_left, pad_top, pad_right, pad_bottom = padding
                    content_h = tensor_h - pad_top - pad_bottom
                    content_w = tensor_w - pad_left - pad_right

                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor.unsqueeze(0).float(),
                        size=(content_h, content_w),
                        mode="nearest",
                    )[0]

                    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                        mask_tensor = torch.nn.functional.pad(
                            mask_tensor,
                            (pad_left, pad_right, pad_top, pad_bottom),
                            mode="constant",
                            value=0,
                        )

                result["masks"] = mask_tensor.to(torch.uint8)

                # Generate per-pixel semantic class labels at the same spatial
                # resolution as masks (required by YOLO26-seg's auxiliary sem
                # loss which indexes sem_masks using instance mask shapes).
                cls_tensor = torch.as_tensor(cls, dtype=torch.float32).squeeze(-1)  # (N,)
                # Per-pixel class = max(class_id * mask_presence); background stays 0.
                sem_masks = (mask_tensor.float() * cls_tensor[:, None, None]).max(0).values  # (H, W)
                result["sem_masks"] = sem_masks
            else:
                result["masks"] = torch.zeros((0, tensor_h, tensor_w), dtype=torch.uint8)
                result["sem_masks"] = torch.zeros((tensor_h, tensor_w), dtype=torch.float32)

        return result
