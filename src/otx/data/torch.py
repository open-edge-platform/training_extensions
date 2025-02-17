"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from .validations import (
    ValidateBatchMixin,
    ValidateItemMixin,
)


@dataclass
class TorchDataItem(ValidateItemMixin):
    """Torch data item implementation."""

    image: torch.Tensor
    label: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    boxes: torch.Tensor | None = None

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
            labels=torch.vstack([item.label for item in items]),
        )


@dataclass
class TorchDataBatch(ValidateBatchMixin):
    """Torch data item batch implementation."""

    images: torch.Tensor
    labels: torch.Tensor | None
    masks: torch.Tensor | None = None
    boxes: torch.Tensor | None = None


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
    labels: torch.Tensor | None
    scores: torch.Tensor | None = None
    feature_vectors: torch.Tensor | None = None
    saliency_maps: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    boxes: torch.Tensor | None = None
