# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from datumaro import Bbox, Image
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.types import OTXTaskType
from otx.types.image import ImageColorChannel

from .base import OTXDataset, Transforms
from .mixins import DataAugSwitchMixin

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset


class OTXDetectionDataset(OTXDataset, DataAugSwitchMixin):  # type: ignore[misc]
    """OTX Dataset for object detection tasks.

    This dataset handles object detection where each image contains multiple objects with
    bounding box annotations. It processes Datumaro dataset items and converts them into
    OTXDataItem format suitable for object detection training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms | None, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        image_color_channel (ImageColorChannel): Color channel format for images (RGB, BGR, etc.).
        stack_images (bool): Whether to stack images in batch processing.
        to_tv_image (bool): Whether to convert images to torchvision format.
        data_format (str): Format of the source data (e.g., "coco", "pascal_voc").

    Example:
        >>> from otx.data.dataset.detection import OTXDetectionDataset
        >>> dataset = OTXDetectionDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     image_color_channel=ImageColorChannel.RGB
        ... )
        >>> item = dataset[0]  # Get first item with bounding boxes
    """

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            image_color_channel=image_color_channel,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        """Get a single data item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            OTXDataItem or None: The processed data item with image, bounding boxes, and labels,
                or None if the item could not be processed.
        """
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

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The object detection task type.
        """
        return OTXTaskType.DETECTION
