# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for OTX data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
import torch.utils._pytree as pytree
from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import (
    Subset,
    bbox_field,
    image_field,
    image_info_field,
    instance_mask_field,
    keypoints_field,
    label_field,
    mask_field,
    subset_field,
)
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask


def register_pytree_node(cls: type[Sample]) -> type[Sample]:
    """Decorator to register an OTX data entity with PyTorch's PyTree.

    This decorator should be applied to every OTX data entity, as TorchVision V2 transforms
    use the PyTree to flatten and unflatten the data entity during runtime.

    Example:
        `MulticlassClsDataEntity` example ::

            @register_pytree_node
            @dataclass
            class MulticlassClsDataEntity(OTXDataEntity):
                ...
    """

    def flatten_fn(obj: object) -> tuple[list[Any], list[str]]:
        obj_dict = dict(obj.__dict__)

        missing_keys = set(obj.__class__.__annotations__.keys()) - set(obj_dict.keys())
        for key in missing_keys:
            obj_dict[key] = getattr(obj, key)

        return (list(obj_dict.values()), list(obj_dict.keys()))

    def unflatten_fn(values: list[Any], context: list[str]) -> object:
        return cls(**dict(zip(context, values)))

    pytree.register_pytree_node(
        cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )
    return cls


@register_pytree_node
class OTXSample(Sample):
    """Base class for OTX data samples."""

    image: np.ndarray | torch.Tensor | tv_tensors.Image | Any
    subset: Subset = subset_field()

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
    def polygons(self) -> np.ndarray | None:
        """Get polygons for the sample."""
        return None

    @property
    def label(self) -> torch.Tensor | None:
        """Optional label property that returns None by default."""
        return None

    @property
    def img_info(self) -> ImageInfo:
        """Get image information for the sample."""
        if self._img_info is None:
            err_msg = "img_info is not set."
            raise ValueError(err_msg)
        return self._img_info

    @img_info.setter
    def img_info(self, value: ImageInfo) -> None:
        self._img_info = value


@register_pytree_node
class ClassificationSample(OTXSample):
    """ClassificationSample is a base class for OTX classification items."""

    subset: Subset = subset_field()

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class ClassificationMultiLabelSample(OTXSample):
    """ClassificationMultiLabelSample is a base class for OTX multi label classification items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: np.ndarray | torch.Tensor = label_field(pl.UInt8(), multi_label=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class ClassificationHierarchicalSample(OTXSample):
    """ClassificationHierarchicalSample is a base class for OTX hierarchical classification items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: np.ndarray | torch.Tensor = label_field(pl.UInt8(), is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class DetectionSample(OTXSample):
    """DetectionSample is a base class for OTX detection items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(
        dtype=pl.UInt8(), format="BGR", channels_first=True
    )
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
    bboxes: np.ndarray | tv_tensors.BoundingBoxes = bbox_field(dtype=pl.Float32())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Convert bboxes to tv_tensors format
        if isinstance(self.bboxes, np.ndarray):
            self.bboxes = tv_tensors.BoundingBoxes(
                self.bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class SegmentationSample(OTXSample):
    """OTXDataItemSample is a base class for OTX data items."""

    subset: Subset = subset_field()
    image: np.ndarray | tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=False)
    masks: tv_tensors.Mask = mask_field(dtype=pl.UInt8(), channels_first=True, has_channels_dim=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)
        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class InstanceSegmentationSample(OTXSample):
    """OTXSample for instance segmentation tasks."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    bboxes: np.ndarray | tv_tensors.BoundingBoxes = bbox_field(dtype=pl.Float32())
    masks: tv_tensors.Mask = instance_mask_field(dtype=pl.UInt8())
    label: torch.Tensor = label_field(is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Convert bboxes to tv_tensors format
        if isinstance(self.bboxes, np.ndarray):
            self.bboxes = tv_tensors.BoundingBoxes(
                self.bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class KeypointSample(OTXSample):
    """KeypointSample is a base class for OTX keypoint detection items."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
    keypoints: torch.Tensor = keypoints_field()
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )
