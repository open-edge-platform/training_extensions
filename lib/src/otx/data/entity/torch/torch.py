# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-specific data item implementations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torchvision.transforms.v2.functional as F  # noqa: N812
from torchvision import tv_tensors

from otx.data.entity.utils import register_pytree_node

from .validations import (
    ValidateBatchMixin,
    ValidateItemMixin,
)

if TYPE_CHECKING:
    import numpy as np
    from datumaro import Polygon
    from torchvision.tv_tensors import BoundingBoxes, Mask

    from otx.data.entity.base import ImageInfo


# NOTE: register_pytree_node and Mapping are required for torchvision.transforms.v2 to work with OTXDataEntity
# TODO(ashwinvaidya17): Remove this once custom transforms are removed
@register_pytree_node
@dataclass
class OTXDataItem(ValidateItemMixin, Mapping):
    """OTX data item implementation.

    Attributes:
        image (torch.Tensor | np.ndarray ): The image tensor
        label (torch.Tensor | None): The label tensor, optional.
        masks (Mask | None): The masks, optional.
        bboxes (BoundingBoxes | None): The bounding boxes, optional.
        keypoints (torch.Tensor | None): The keypoints, optional.
        polygons (list[Polygon] | None): The polygons, optional.
        img_info (ImageInfo | None): Additional image information, optional.
    """

    image: torch.Tensor | np.ndarray
    label: torch.Tensor | None = None
    masks: Mask | None = None
    bboxes: BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None
    polygons: list[Polygon] | None = None
    img_info: ImageInfo | None = None  # TODO(ashwinvaidya17): revisit and try to remove this

    @staticmethod
    def collate_fn(items: list[OTXDataItem]) -> OTXDataBatch:
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

        return OTXDataBatch(
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

    def to_tv_image(self) -> OTXDataItem:
        """Return a new instance with the `image` attribute converted to a TorchVision Image if it is a NumPy array.

        Returns:
            A new instance with the `image` attribute converted to a TorchVision Image, if applicable.
            Otherwise, return this instance as is.
        """
        if isinstance(self.image, tv_tensors.Image):
            return self

        return self.wrap(image=F.to_image(self.image))

    def wrap(self, **kwargs) -> OTXDataItem:
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
class OTXDataBatch(ValidateBatchMixin):
    """Torch data item batch implementation."""

    batch_size: int  # TODO(ashwinvaidya17): Remove this
    images: torch.Tensor | list[torch.Tensor]
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    polygons: list[list[Polygon]] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None  # TODO(ashwinvaidya17): revisit

    def pin_memory(self) -> OTXDataBatch:
        """Pin memory for member tensor variables."""
        # https://github.com/pytorch/pytorch/issues/116403

        kwargs = {}

        def maybe_pin(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, torch.Tensor):
                return x.pin_memory()
            return x

        def maybe_wrap_tv(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, tv_tensors.TVTensor):
                return tv_tensors.wrap(x.pin_memory(), like=x)
            return maybe_pin(x)

        # Handle images separately because of tv_tensors wrapping
        if self.images is not None:
            if isinstance(self.images, list):
                kwargs["images"] = [maybe_wrap_tv(img) for img in self.images]
            else:
                kwargs["images"] = maybe_wrap_tv(self.images)

        # Generic handler for all other fields
        for field in ["labels", "bboxes", "keypoints", "masks"]:
            value = getattr(self, field)
            if value is not None:
                kwargs[field] = [maybe_wrap_tv(v) if v is not None else None for v in value]

        return self.wrap(**kwargs)

    def wrap(self, **kwargs) -> OTXDataBatch:
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
class OTXPredItem(OTXDataItem):
    """Torch prediction data item implementation."""

    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None


@dataclass
class OTXPredBatch(OTXDataBatch):
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
