# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter: VisionDataset sample -> Ultralytics sample dict."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.base import ImageInfo

from .geometry import build_ratio_pad, xyxy_abs_to_xywh_norm


class UltralyticsDatasetAdapter(torch.utils.data.Dataset):
    """Wraps a getitune ``VisionDataset`` and yields Ultralytics sample dicts.

    Does NOT apply any additional augmentation — the upstream
    ``VisionDataset`` already produces float32 CHW ``[0, 1]`` samples.
    """

    def __init__(self, vision_dataset: VisionDataset, *, include_masks: bool = False) -> None:
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

        # --- Geometry from ImageInfo ---
        img_info: ImageInfo | None = getattr(sample, "img_info", None)
        if img_info is not None:
            ori_shape = img_info.ori_shape
            # img_shape is the resized size *before* padding.
            resized_shape = img_info.img_shape
            padding = img_info.padding
        else:
            ori_shape = (tensor_h, tensor_w)
            resized_shape = (tensor_h, tensor_w)
            padding = (0, 0, 0, 0)

        # Ultralytics validators use ratio_pad for bbox postprocessing.
        ratio_pad = build_ratio_pad(ori_shape, resized_shape, padding)

        # --- Bounding boxes (XYXY absolute -> XYWH normalised) ---
        # Ultralytics expects centre-xywh normalised by the model input
        # (tensor) dimensions.  getitune's DataModule may deliver bboxes in
        # the *original* image coordinate space when ``resize_targets=False``
        # (typical for val/test subsets).  Detect this via ``canvas_size``
        # on ``tv_tensors.BoundingBoxes`` and rescale to the tensor space.
        bboxes_raw = getattr(sample, "bboxes", None)
        if bboxes_raw is not None and len(bboxes_raw) > 0:
            bboxes_for_norm = bboxes_raw
            # When canvas_size differs from the tensor dims the bboxes are
            # still in the original (or some other non-tensor) coord space.
            canvas = getattr(bboxes_raw, "canvas_size", None)
            if canvas is not None and tuple(canvas) != (tensor_h, tensor_w):
                bboxes_np = bboxes_raw.detach().cpu().numpy().astype(np.float32).copy()
                # Use scale_factor + padding from ImageInfo for correct
                # letterbox transformation (simple ratio is wrong when
                # keep_aspect_ratio padding is present).
                if img_info is not None and img_info.scale_factor is not None:
                    scale_h, scale_w = img_info.scale_factor
                    pad_left, pad_top = padding[0], padding[1]
                    bboxes_np[:, 0::2] = bboxes_np[:, 0::2] * scale_w + pad_left
                    bboxes_np[:, 1::2] = bboxes_np[:, 1::2] * scale_h + pad_top
                else:
                    canvas_h, canvas_w = canvas
                    bboxes_np[:, 0::2] *= tensor_w / canvas_w
                    bboxes_np[:, 1::2] *= tensor_h / canvas_h
                bboxes_for_norm = bboxes_np
            bboxes_xywh = xyxy_abs_to_xywh_norm(bboxes_for_norm, img_w=tensor_w, img_h=tensor_h)
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
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

        # --- Instance masks (for segmentation) ---
        if self._include_masks:
            masks_raw = getattr(sample, "masks", None)
            if masks_raw is not None and len(masks_raw) > 0:
                mask_tensor = masks_raw
                if not isinstance(mask_tensor, torch.Tensor):
                    mask_tensor = torch.as_tensor(mask_tensor)
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                # Resize masks to tensor dimensions if they differ (val/test
                # subsets may deliver masks at original image resolution while
                # the image tensor is already resized + padded).
                if mask_tensor.shape[1:] != (tensor_h, tensor_w):
                    mask_tensor = (
                        torch.nn.functional.interpolate(
                            mask_tensor.unsqueeze(0).float(),
                            size=(tensor_h, tensor_w),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        .gt_(0.5)
                        .float()
                    )
                result["masks"] = mask_tensor.float()

                # Generate per-pixel semantic class labels at the same spatial
                # resolution as masks (required by YOLO26-seg's auxiliary sem
                # loss which indexes sem_masks using instance mask shapes).
                cls_tensor = torch.as_tensor(cls, dtype=torch.float32).squeeze(-1)  # (N,)
                # Per-pixel class = max(class_id * mask_presence); background stays 0.
                sem_masks = (mask_tensor * cls_tensor[:, None, None]).max(0).values  # (H, W)
                result["sem_masks"] = sem_masks
            else:
                result["masks"] = torch.zeros((0, tensor_h, tensor_w), dtype=torch.float32)
                result["sem_masks"] = torch.zeros((tensor_h, tensor_w), dtype=torch.float32)

        return result
