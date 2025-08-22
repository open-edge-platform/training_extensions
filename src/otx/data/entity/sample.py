# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for OTX data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from datumaro import Mask
from datumaro.components.media import Image
from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import ImageInfo, label_field, image_field

if TYPE_CHECKING:
    from datumaro import Polygon, DatasetItem
    from torchvision.tv_tensors import BoundingBoxes, Mask


class ClassificationSample(Sample):
    """OTXDataItemSample is a base class for OTX data items."""
    label: int = label_field(pl.Int32())
    image: torch.Tensor | np.ndarray = image_field(dtype=pl.UInt8)

    @classmethod
    def from_dm_item(cls, item: DatasetItem) -> "ClassificationSample":
        """
        Create a ClassificationSample from a Datumaro DatasetItem.

        Args:
            item: Datumaro DatasetItem containing image and label

        Returns:
            ClassificationSample: Instance with image and label set
        """
        image = item.media_as(Image).data
        label = item.annotations[0].label if item.annotations else None
        return cls(image=image, label=label)

    @property
    def masks(self) -> Mask | None:
        return None

    @property
    def bboxes(self) -> BoundingBoxes | None:
        return None

    @property
    def keypoints(self) -> torch.Tensor | None:
        return None

    @property
    def polygons(self) -> list[Polygon] | None:
        return None

    @property
    def img_info(self) -> ImageInfo | None:
        return None
