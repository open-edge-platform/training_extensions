# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for OTXDataset using new Datumaro experimental Dataset."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, Iterable, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from otx import LabelInfo, NullLabelInfo

if TYPE_CHECKING:
    from datumaro.experimental import Dataset

from otx.data.entity.sample import OTXSample
from otx.data.entity.torch.torch import OTXDataBatch
from otx.data.transform_libs.torchvision import Compose
from otx.types.image import ImageColorChannel

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


def _default_collate_fn(items: list[OTXSample]) -> OTXDataBatch:
    """Collate OTXSample items into an OTXDataBatch.

    Args:
        items: List of OTXSample items to batch
    Returns:
        Batched OTXSample items with stacked tensors
    """
    # Convert images to float32 tensors before stacking
    image_tensors = []
    for item in items:
        img = item.image
        if isinstance(img, torch.Tensor):
            # Convert to float32 if not already
            if img.dtype != torch.float32:
                img = img.float()
        else:
            # Convert numpy array to float32 tensor
            img = torch.from_numpy(img).float()
        image_tensors.append(img)

    # Try to stack images if they have the same shape
    if len(image_tensors) > 0 and all(t.shape == image_tensors[0].shape for t in image_tensors):
        images = torch.stack(image_tensors)
    else:
        images = image_tensors

    return OTXDataBatch(
        batch_size=len(items),
        images=images,
        labels=[item.label for item in items] if items[0].label is not None else None,
        masks=[item.masks for item in items] if any(item.masks is not None for item in items) else None,
        bboxes=[item.bboxes for item in items] if any(item.bboxes is not None for item in items) else None,
        keypoints=[item.keypoints for item in items] if any(item.keypoints is not None for item in items) else None,
        polygons=[item.polygons for item in items if item.polygons is not None]
        if any(item.polygons is not None for item in items)
        else None,
        imgs_info=[item.img_info for item in items] if any(item.img_info is not None for item in items) else None,
    )


class OTXDataset(TorchDataset):
    """Base OTXDataset using new Datumaro experimental Dataset.

    Defines basic logic for OTX datasets.

    Args:
        transforms: Transforms to apply on images
        image_color_channel: Color channel of images
        stack_images: Whether or not to stack images in collate function in OTXBatchData entity.
        sample_type: Type of sample to use for this dataset
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
        sample_type: type[OTXSample] = OTXSample,
    ) -> None:
        self.transforms = transforms
        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        self.sample_type = sample_type
        self.max_refetch = max_refetch
        self.data_format = data_format
        if (
            hasattr(dm_subset, "schema")
            and hasattr(dm_subset.schema, "attributes")
            and "label" in dm_subset.schema.attributes
        ):
            labels = dm_subset.schema.attributes["label"].categories.labels
            self.label_info = LabelInfo(
                label_names=labels,
                label_groups=[labels],
                label_ids=[str(i) for i in range(len(labels))],
            )
        else:
            self.label_info = NullLabelInfo()
        self.dm_subset = dm_subset

    def __len__(self) -> int:
        return len(self.dm_subset)

    def _sample_another_idx(self) -> int:
        return np.random.randint(0, len(self))

    def _apply_transforms(self, entity: OTXSample) -> OTXSample | None:
        if isinstance(self.transforms, Compose):
            if self.to_tv_image:
                entity.as_tv_image()
            return self.transforms(entity)
        if isinstance(self.transforms, Iterable):
            return self._iterable_transforms(entity)
        if callable(self.transforms):
            return self.transforms(entity)
        return None

    def _iterable_transforms(self, item: OTXSample) -> OTXSample | None:
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

    def __getitem__(self, index: int) -> OTXSample:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = self._sample_another_idx()

        msg = f"Reach the maximum refetch number ({self.max_refetch})"
        raise RuntimeError(msg)

    def _get_item_impl(self, index: int) -> OTXSample | None:
        dm_item = self.dm_subset[index]
        return self._apply_transforms(dm_item)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect samples into a batch in data loader."""
        return _default_collate_fn

    @abc.abstractmethod
    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int, list[int]]:
        """Get a dictionary with class labels as keys and lists of corresponding sample indices as values."""
