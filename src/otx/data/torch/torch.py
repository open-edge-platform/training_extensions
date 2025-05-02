"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, asdict
from typing import TYPE_CHECKING, Any, Sequence

import torch

from otx.core.data.entity.utils import register_pytree_node

from .validations import (
    ValidateBatchMixin,
    ValidateItemMixin,
)

from torchvision import tv_tensors

if TYPE_CHECKING:
    from datumaro import Polygon
    from torchvision.tv_tensors import BoundingBoxes, Mask

    from otx.core.data.entity.base import ImageInfo


# NOTE: register_pytree_node and Mapping are required for torchvision.transforms.v2 to work with OTXDataEntity
# TODO(ashwinvaidya17): Remove this once custom transforms are removed
@register_pytree_node
@dataclass
class TorchDataItem(ValidateItemMixin, Mapping):
    """Torch data item implementation.

    Attributes:
        image (torch.Tensor): The image tensor.
        label (torch.Tensor | None): The label tensor, optional.
        masks (Mask | None): The masks, optional.
        bboxes (BoundingBoxes | None): The bounding boxes, optional.
        keypoints (torch.Tensor | None): The keypoints, optional.
        polygons (list[Polygon] | None): The polygons, optional.
        img_info (ImageInfo | None): Additional image information, optional.
    """

    image: torch.Tensor
    label: torch.Tensor | None = None
    masks: Mask | None = None
    bboxes: BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None
    polygons: list[Polygon] | None = None
    img_info: ImageInfo | None = None  # TODO(ashwinvaidya17): revisit and try to remove this

    @staticmethod
    def collate_fn(items: list[TorchDataItem]) -> TorchDataBatch:
        """Collate TorchDataItems into a batch.

        Args:
            items: List of TorchDataItems to batch
        Returns:
            Batched TorchDataItems with stacked tensors
        """
        # Check if all images have the same size. TODO(kprokofi): remove this check once OV IR models are moved.
        if all(item.image.shape == items[0].image.shape for item in items):
            images = torch.stack([item.image for item in items])
        else:
            # we need this only in case of OV inference, where no resize
            images = [item.image for item in items]

        return TorchDataBatch(
            batch_size=len(items),
            images=images,
            labels=[item.label for item in items],
            bboxes=[item.bboxes for item in items],
            keypoints=[item.keypoints for item in items],
            masks=[item.masks for item in items],
            polygons=[item.polygons for item in items],  # type: ignore[misc]
            imgs_info=[item.img_info for item in items],
        )

    def __iter__(self) -> Iterator[str]:
        for field_ in fields(self):
            yield field_.name

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        return getattr(self, key)

    def __len__(self) -> int:
        return len(fields(self))


@dataclass
class TorchDataBatch(ValidateBatchMixin):
    """Torch data item batch implementation."""

    batch_size: int  # TODO(ashwinvaidya17): Remove this
    images: torch.Tensor | list[torch.Tensor]
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    polygons: list[list[Polygon]] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None  # TODO(ashwinvaidya17): revisit

    def pin_memory(self):
        """Pin memory for member tensor variables."""
        # TODO(vinnamki): Keep track this issue
        # https://github.com/pytorch/pytorch/issues/116403

        return self.wrap(
            images=(
                [tv_tensors.wrap(image.pin_memory(), like=image) for image in self.images]
                if isinstance(self.images, list)
                else tv_tensors.wrap(self.images.pin_memory(), like=self.images)
            ),
            bboxes=[tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes],
            labels=[label.pin_memory() for label in self.labels],
        )
    
    def wrap(self, **kwargs):
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass
        Returns:
            Updated dataclass
        """
        updated_kwargs = asdict(self)
        updated_kwargs.update(**kwargs)
        return self.__class__(**updated_kwargs)


@dataclass
class TorchPredItem(TorchDataItem):
    """Torch prediction data item implementation."""

    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None


@dataclass
class TorchPredBatch(TorchDataBatch):
    """Torch prediction data item batch implementation."""

    scores: list[torch.Tensor] | None = None
    feature_vector: list[torch.Tensor] | None = None
    saliency_map: list[torch.Tensor] | None = None

    @property
    def has_xai_outputs(self) -> bool:
        """Check if the batch has XAI outputs.

        Necessary for compatibility with tests.
        """
        # TODO(ashwinvaidya17): the tests should directly refer to saliency map.
        return self.saliency_map is not None and len(self.saliency_map) > 0
