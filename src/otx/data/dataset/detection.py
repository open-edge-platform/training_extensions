# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

import numpy as np
import torch
from datumaro import Bbox, Image
from torchvision import tv_tensors

from otx.algo.callbacks.aug_scheduler import DataAugSwitch
from otx.core.data.entity.base import ImageInfo
from otx.data import OTXDataItem

from .base import OTXDataset


class OTXDetectionDataset(OTXDataset):
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

        if hasattr(self, "data_aug_switch") and isinstance(self.data_aug_switch, DataAugSwitch):
            # Set the shared epoch for data augmentation
            self.to_tv_image, self.transforms = self.data_aug_switch.current_transforms
        return self._apply_transforms(entity)

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set data augmentation switch."""
        self.data_aug_switch = data_aug_switch
