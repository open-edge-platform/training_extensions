"""Datumaro to torch resolvers."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from datumaro import Bbox, Points
from torchvision import tv_tensors

from otx.core.types.label import LabelInfo

if TYPE_CHECKING:
    from datumaro import Annotation, DatasetItem


class DatumaroResolver:
    """Resolver for datumaro annotations to torch tensors."""

    @staticmethod
    def resolve_label(item: DatasetItem, from_type: type[Annotation]) -> torch.Tensor:
        """Resolve labels from item."""
        label_anns = [ann for ann in item.annotations if isinstance(ann, from_type)]
        return torch.as_tensor([ann.label for ann in label_anns], dtype=torch.long)

    @staticmethod
    def resolve_bbox(item: DatasetItem, img_shape: tuple[int, int]) -> tv_tensors.BoundingBoxes:
        """Resolve bboxes from item."""
        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        return tv_tensors.BoundingBoxes(
            bboxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_shape,
            dtype=torch.float32,
        )

    @staticmethod
    def resolve_keypoints(item: DatasetItem, label_info: LabelInfo) -> torch.Tensor:
        """Resolve keypoints from item."""
        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        keypoint_anns = [ann for ann in item.annotations if isinstance(ann, Points)]
        keypoints = (
            np.stack([ann.points for ann in keypoint_anns], axis=0).astype(np.float32)
            if len(keypoint_anns) > 0
            else np.zeros((0, len(label_info.label_names) * 2), dtype=np.float32)
        ).reshape(-1, 2)

        keypoints_visible = (
            (torch.tensor([ann.visibility for ann in keypoint_anns], dtype=torch.int8) > 1).reshape(-1)
            if len(keypoint_anns) > 0 and hasattr(keypoint_anns[0], "visibility")
            else torch.minimum(1, keypoints)[..., 0]
        )
        return torch.cat([torch.tensor(keypoints, dtype=torch.float32), keypoints_visible.reshape(-1, 1)], dim=1)
