# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for OTX data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from datumaro import Mask
from datumaro.components.media import Image
from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import image_field, image_info_field, label_field, mask_field
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo

if TYPE_CHECKING:
    from datumaro import DatasetItem, Polygon
    from torchvision.tv_tensors import BoundingBoxes, Mask


class OTXSample(Sample):
    """Base class for OTX data samples."""

    image: np.ndarray | torch.Tensor | tv_tensors.Image | Any

    def as_tv_image(self) -> None:
        """Convert image to torchvision tv_tensors Image format."""
        if isinstance(self.image, tv_tensors.Image):
            return
        if isinstance(self.image, (np.ndarray, torch.Tensor)):
            self.image = tv_tensors.Image(self.image)
            return
        msg = "OTXSample must have an image"
        raise ValueError(msg)

    @property
    def masks(self) -> Mask | None:
        """Get masks for the sample."""
        return None

    @property
    def bboxes(self) -> BoundingBoxes | None:
        """Get bounding boxes for the sample."""
        return None

    @property
    def keypoints(self) -> torch.Tensor | None:
        """Get keypoints for the sample."""
        return None

    @property
    def polygons(self) -> list[Polygon] | None:
        """Get polygons for the sample."""
        return None

    @property
    def label(self) -> torch.Tensor | None:
        """Optional label property that returns None by default."""
        return None

    @property
    def img_info(self) -> ImageInfo | None:
        """Get image information for the sample."""
        if getattr(self, "_img_info", None) is None:
            image = getattr(self, "image", None)
            if image is not None and hasattr(image, "shape") and len(image.shape) == 3:
                img_shape = image.shape[:2]
            else:
                return None
            self._img_info = ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=img_shape,
            )
        return self._img_info

    @img_info.setter
    def img_info(self, value: ImageInfo) -> None:
        self._img_info = value


class ClassificationSample(OTXSample):
    """OTXDataItemSample is a base class for OTX data items."""

    image: np.ndarray | tv_tensors.Image = image_field(dtype=pl.UInt8)
    label: torch.Tensor = label_field(pl.Int32())

    @classmethod
    def from_dm_item(cls, item: DatasetItem) -> ClassificationSample:
        """Create a ClassificationSample from a Datumaro DatasetItem.

        Args:
            item: Datumaro DatasetItem containing image and label

        Returns:
            ClassificationSample: Instance with image and label set
        """
        image = item.media_as(Image).data
        label = item.annotations[0].label if item.annotations else None

        img_shape = image.shape[:2]
        img_info = ImageInfo(
            img_idx=0,
            img_shape=img_shape,
            ori_shape=img_shape,
        )

        sample = cls(
            image=image,
            label=torch.as_tensor(label, dtype=torch.long) if label is not None else torch.tensor(-1, dtype=torch.long),
        )
        sample.img_info = img_info
        return sample


class SegmentationSample(OTXSample):
    """OTXDataItemSample is a base class for OTX data items."""

    image: np.ndarray | tv_tensors.Image = image_field(dtype=pl.UInt8)
    masks: np.ndarray | tv_tensors.Mask = mask_field(dtype=pl.UInt8)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)
        self.image = tv_tensors.Image(self.image.transpose(2, 0, 1))
        self.masks = tv_tensors.Mask(self.masks[np.newaxis, ...])
        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )
