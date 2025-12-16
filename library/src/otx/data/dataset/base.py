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
from otx.types import OTXTaskType

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]

RNG = np.random.default_rng(42)


def _ensure_chw_format(img: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor is in CHW format with 3 channels.

    Args:
        img: Image tensor that may be in HWC or CHW format

    Returns:
        Image tensor in CHW format (C, H, W) for 3D or (B, C, H, W) for 4D with 3 channels
    """
    # Handle 2D grayscale images (H, W) - add channel dimension
    if img.ndim == 2:
        img = img.unsqueeze(0)  # (H, W) -> (1, H, W)
        img = img.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
    elif img.ndim == 3:
        # Check if last dimension is likely channels (small value like 1, 3, or 4)
        # and first dimension is not (larger, like image height)
        if img.shape[-1] in (1, 3, 4) and img.shape[0] > 4:
            # HWC format detected, convert to CHW
            img = img.permute(2, 0, 1)
        # If 4 channels (RGBA), convert to 3 channels (RGB)
        if img.shape[0] == 4:
            img = img[:3]
        # If 1 channel (grayscale/palette), convert to 3 channels (RGB)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
    return img


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
        # Ensure image is in CHW format
        img = _ensure_chw_format(img)
        image_tensors.append(img)

    # Try to stack images if they have the same shape
    if len(image_tensors) > 0 and all(t.shape == image_tensors[0].shape for t in image_tensors):
        images = torch.stack(image_tensors)
        # Safety: ensure stacked tensor is BCHW. If it's in BHWC or BHCW, fix it.
        if images.ndim == 4:
            # BHWC -> BCHW
            if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)
            # BHCW -> BCHW (channels at dim=2)
            elif images.shape[2] in (1, 3) and images.shape[1] not in (1, 3):
                images = images.permute(0, 2, 1, 3)
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

    This class defines the basic logic and interface for OTX datasets, providing
    functionality for data transformation, image decoding, and label handling.

    Args:
        dm_subset (DmDataset): Datumaro subset of a dataset.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int, optional): Maximum number of times to attempt fetching a valid image. Defaults to 1000.
        stack_images (bool, optional): Whether to stack images in the collate function in OTXBatchData entity.
            Defaults to True.
        to_tv_image (bool, optional): Whether to convert images to TorchVision format. Defaults to True.
        data_format (str, optional): Source data format originally passed to Datumaro (e.g., "arrow"). Defaults to "".

    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
        sample_type: type[OTXSample] = OTXSample,
    ) -> None:
        self.transforms = transforms
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        self.sample_type = sample_type
        self.max_refetch = max_refetch
        self.data_format = data_format
        self.label_info: LabelInfo = NullLabelInfo()
        self.dm_subset = dm_subset

    def __len__(self) -> int:
        return len(self.dm_subset)

    def _apply_transforms(self, entity: OTXSample) -> OTXSample | None:
        if self.transforms is None:
            return entity
        if isinstance(self.transforms, Compose):
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

            index = RNG.integers(0, len(self))

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
    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels to lists of corresponding sample indices.

        Args:
            use_string_label (bool, optional): If True, use string labels as keys; otherwise, use integer labels.

        Returns:
            dict[int | str, list[int]]: A dictionary where each key is a class label (int or str) and each value is a
            list of sample indices belonging to that class.
        """

    @property
    def task_type(self) -> OTXTaskType | None:
        """OTX Task Type for the dataset. Can be None if no task is defined."""
        return None
