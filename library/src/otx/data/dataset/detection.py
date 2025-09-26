# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

import numpy as np
import torch
from datumaro import Bbox, Image
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem

from .base import OTXDataset
from .mixins import DataAugSwitchMixin


class OTXDetectionDataset(OTXDataset, DataAugSwitchMixin):  # type: ignore[misc]
    """OTXDataset class for detection task."""

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = OTXDataItem(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
                dtype=torch.float32,
            ),
            label=torch.as_tensor([ann.label for ann in bbox_anns], dtype=torch.long),
        )
        # Apply augmentation switch if available
        if self.has_dynamic_augmentation:
            self._apply_augmentation_switch()

        return self._apply_transforms(entity)
