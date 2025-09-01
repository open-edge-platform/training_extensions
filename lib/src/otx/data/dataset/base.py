# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for OTXDataset."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Union

import cv2
import numpy as np
from datumaro.components.annotation import AnnotationType
from datumaro.util.image import IMAGE_BACKEND, IMAGE_COLOR_CHANNEL, ImageBackend
from datumaro.util.image import ImageColorChannel as DatumaroImageColorChannel
from torch.utils.data import Dataset

from otx.data.entity.torch import OTXDataItem
from otx.data.transform_libs.torchvision import Compose
from otx.types.image import ImageColorChannel
from otx.types.label import LabelInfo, NullLabelInfo

if TYPE_CHECKING:
    from datumaro import DatasetSubset, Image


Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


@contextmanager
def image_decode_context() -> Iterator[None]:
    """Change Datumaro image decode context.

    Use PIL Image decode because of performance issues.
    With this context, `dm.Image.data` will return BGR numpy image tensor.
    """
    ori_image_backend = IMAGE_BACKEND.get()
    ori_image_color_scale = IMAGE_COLOR_CHANNEL.get()

    IMAGE_BACKEND.set(ImageBackend.PIL)
    # TODO(vinnamki): This should be changed to
    # if to_rgb:
    #     IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_RGB)
    # else:
    #     IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_BGR)
    # after merging https://github.com/openvinotoolkit/datumaro/pull/1501
    IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_RGB)

    yield

    IMAGE_BACKEND.set(ori_image_backend)
    IMAGE_COLOR_CHANNEL.set(ori_image_color_scale)


class OTXDataset(Dataset):
    """Base OTXDataset.

    Defines basic logic for OTX datasets.

    Args:
        dm_subset: Datumaro subset of a dataset
        transforms: Transforms to apply on images
        max_refetch: Maximum number of images to fetch in cache
        image_color_channel: Color channel of images
        stack_images: Whether or not to stack images in collate function in OTXBatchData entity.
        data_format: Source data format, which was originally passed to datumaro (could be arrow for instance).

    """

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        self.dm_subset = dm_subset
        self.transforms = transforms
        self.max_refetch = max_refetch
        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        self.data_format = data_format

        if self.dm_subset.categories() and data_format == "arrow":
            self.label_info = LabelInfo.from_dm_label_groups_arrow(self.dm_subset.categories()[AnnotationType.label])
        elif self.dm_subset.categories():
            self.label_info = LabelInfo.from_dm_label_groups(self.dm_subset.categories()[AnnotationType.label])
        else:
            self.label_info = NullLabelInfo()

    def __len__(self) -> int:
        return len(self.dm_subset)

    def _sample_another_idx(self) -> int:
        return np.random.default_rng().integers(0, len(self))

    def _apply_transforms(self, entity: OTXDataItem) -> OTXDataItem | None:
        if isinstance(self.transforms, Compose):
            if self.to_tv_image:
                entity = entity.to_tv_image()
            return self.transforms(entity)
        if isinstance(self.transforms, Iterable):
            return self._iterable_transforms(entity)
        if callable(self.transforms):
            return self.transforms(entity)

        raise TypeError(self.transforms)

    def _iterable_transforms(self, item: OTXDataItem) -> OTXDataItem | None:
        if not isinstance(self.transforms, list):
            raise TypeError(item)

        results = item
        for transform in self.transforms:
            results = transform(results)
            # MMCV transform can produce None. Please see
            # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            if results is None:
                return None

        return results

    def __getitem__(self, index: int) -> OTXDataItem:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = self._sample_another_idx()

        msg = f"Reach the maximum refetch number ({self.max_refetch})"
        raise RuntimeError(msg)

    def _get_img_data_and_shape(
        self,
        img: Image,
        roi: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int], dict[str, Any] | None]:
        """Get image data and shape.

        This method is used to get image data and shape from Datumaro image object.
        If ROI is provided, the image data is extracted from the ROI.

        Args:
            img (Image): Image object from Datumaro.
            roi (dict[str, Any] | None, Optional): Region of interest.
                Represented by dict with coordinates and some meta information.

        Returns:
                The image data, shape, and ROI meta information
        """
        roi_meta = None

        with image_decode_context():
            img_data = (
                img.data
                if self.image_color_channel == ImageColorChannel.RGB
                else cv2.cvtColor(img.data, cv2.COLOR_RGB2BGR)
            )

        if img_data is None:
            msg = "Cannot get image data"
            raise RuntimeError(msg)

        if roi and isinstance(roi, dict):
            # extract ROI from image
            shape = roi["shape"]
            h, w = img_data.shape[:2]
            x1, y1, x2, y2 = (
                int(np.clip(np.trunc(shape["x1"] * w), 0, w)),
                int(np.clip(np.trunc(shape["y1"] * h), 0, h)),
                int(np.clip(np.ceil(shape["x2"] * w), 0, w)),
                int(np.clip(np.ceil(shape["y2"] * h), 0, h)),
            )
            if (x2 - x1) * (y2 - y1) <= 0:
                msg = f"ROI has zero or negative area. ROI coordinates: {x1}, {y1}, {x2}, {y2}"
                raise ValueError(msg)

            img_data = img_data[y1:y2, x1:x2]
            roi_meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "orig_image_shape": (h, w)}

        return img_data, img_data.shape[:2], roi_meta

    @abstractmethod
    def _get_item_impl(self, idx: int) -> OTXDataItem | None:
        pass

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect KeypointDetDataEntity into KeypointDetBatchDataEntity in data loader."""
        return OTXDataItem.collate_fn
