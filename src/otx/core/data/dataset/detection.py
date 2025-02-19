# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from datumaro import Bbox, Image
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.core.data.entity.base import ImageInfo
from otx.data.torch import TorchDataItem

from .base import OTXDataset


class OTXDetectionDataset(OTXDataset):
    """OTXDataset class for detection task."""

    def _get_item_impl(self, index: int) -> TorchDataItem | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape, _ = self._get_img_data_and_shape(img)
        image = to_dtype(to_image(img_data), dtype=torch.float32) / 255.0
        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        boxes = tv_tensors.BoundingBoxes(
            bboxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_shape,
            dtype=torch.float32,
        )
        image, boxes = self.transforms(image, boxes)

        return TorchDataItem(
            image=image,
            boxes=boxes,
            label=torch.as_tensor([ann.label for ann in bbox_anns], dtype=torch.long),
            imgs_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
        )

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return TorchDataItem.collate_fn
