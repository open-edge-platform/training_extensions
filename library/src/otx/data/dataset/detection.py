# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import torch
from datumaro import Bbox, Image
from torchvision import tv_tensors

from otx.types.image import ImageColorChannel
from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.types import OTXTaskType

from .base import OTXDataset, Transforms
from .mixins import DataAugSwitchMixin

if TYPE_CHECKING:
    from datumaro import DatasetSubset


class OTXDetectionDataset(OTXDataset, DataAugSwitchMixin):  # type: ignore[misc]
    """OTXDataset class for detection task."""
    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        task_type: OTXTaskType = OTXTaskType.DETECTION,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(dm_subset=dm_subset,
                         task_type=task_type,
                         transforms=transforms,
                         max_refetch=max_refetch, image_color_channel=image_color_channel, stack_images=stack_images, to_tv_image=to_tv_image, data_format=data_format)

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
