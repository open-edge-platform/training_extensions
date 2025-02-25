"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .validations import (
    ValidateBatchMixin,
    ValidateItemMixin,
)

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask

    from otx.core.data.entity.base import ImageInfo


@dataclass
class TorchDataItem(ValidateItemMixin):
    """Torch data item implementation."""

    image: torch.Tensor
    label: torch.Tensor | None = None
    mask: Mask | None = None
    boxes: BoundingBoxes | None = None
    imgs_info: ImageInfo | None = None  # TODO(ashwinvaidya17): revisit and try to remove this

    @staticmethod
    def collate_fn(items: list[TorchDataItem]) -> TorchDataBatch:
        """Collate TorchDataItems into a batch.

        Args:
            items: List of TorchDataItems to batch
        Returns:
            Batched TorchDataItems with stacked tensors
        """
        return TorchDataBatch(
            images=torch.stack([item.image for item in items]),
            labels=[item.label for item in items],
            boxes=[item.boxes for item in items],
            masks=[item.mask for item in items],
            imgs_infos=[item.imgs_info for item in items],
        )


@dataclass
class TorchDataBatch(ValidateBatchMixin):
    """Torch data item batch implementation."""

    images: torch.Tensor
    labels: list[torch.Tensor] | None
    masks: list[Mask] | None = None
    boxes: list[BoundingBoxes] | None = None
    imgs_infos: list[ImageInfo | None] | None = None  # TODO(ashwinvaidya17): revisit


@dataclass
class TorchPredItem(ValidateItemMixin):
    """Torch prediction data item implementation."""

    image: torch.Tensor
    label: torch.Tensor | None
    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None


@dataclass
class TorchPredBatch(ValidateBatchMixin):
    """Torch prediction data item batch implementation."""

    images: torch.Tensor
    labels: list[torch.Tensor] | None
    scores: list[torch.Tensor] | None = None
    feature_vectors: list[torch.Tensor] | None = None
    saliency_maps: list[torch.Tensor] | None = None
    masks: list[torch.Tensor] | None = None
    boxes: list[torch.Tensor] | None = None
    imgs_infos: list[ImageInfo | None] | None = None  # TODO(ashwinvaidya17): revisit

    @property
    def has_xai_outputs(self) -> bool:
        """Check if the batch has XAI outputs.

        Necessary for compatibility with tests.
        """
        # TODO(ashwinvaidya17): the tests should directly refer to saliency maps.
        return self.saliency_maps is not None and len(self.saliency_maps) > 0
