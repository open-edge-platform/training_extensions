"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import DataItem, DataItemBatch, PredItem, PredItemBatch


@dataclass
class TorchDataItem(DataItem[Tensor]):
    """Torch data item implementation.

    Args:
        image: Image tensor of shape (C, H, W) and type float32
        label: Label tensor, either scalar for multi-class or 1D for multi-label
    """

    @staticmethod
    def collate_fn(items: list[TorchDataItem]) -> TorchDataItemBatch:
        """Collate TorchDataItems into a batch.

        Args:
            items: List of TorchDataItems to batch

        Returns:
            Batched TorchDataItems with stacked tensors
        """
        return TorchDataItemBatch(
            images=torch.stack([item.image for item in items]),
            labels=torch.stack([item.label for item in items]),
        )

    def validate_image(self) -> None:
        """Validate image tensor format.

        Raises:
            ValueError: If image is not a 3D float32 tensor with shape (3, H, W)
        """
        if not (
            isinstance(self.image, Tensor)
            and self.image.ndim == 3
            and self.image.shape[0] == 3
            and self.image.dtype == torch.float32
        ):
            msg = "Image must be a float32 tensor with shape (3, H, W)"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label tensor format.

        Raises:
            ValueError: If label is not a scalar or 1D long tensor
        """
        if not isinstance(self.label, Tensor):
            msg = "Label must be a torch.Tensor"
            raise TypeError(msg)

        if not (
            (self.label.ndim == 0 and self.label.dtype == torch.long)  # Multi-class
            or (self.label.ndim == 1 and self.label.dtype == torch.long)  # Multi-label/hierarchical
        ):
            msg = "Label must be a scalar or 1D long tensor"
            raise ValueError(msg)


@dataclass
class TorchDataItemBatch(DataItemBatch[Tensor]):
    """Torch data item batch implementation."""

    def validate_image(self) -> None:
        """Validate image tensor format."""
        if not (isinstance(self.images, Tensor) and self.images.ndim == 4):
            msg = f"Image must have shape (B, C, H, W), but got {self.images.shape}"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label tensor format."""
        if self.labels.ndim != 2:
            msg = f"Label must have shape (B, ), but got {self.labels.shape}"
            raise ValueError(msg)


@dataclass
class TorchPredItem(PredItem[Tensor]):
    """Torch prediction data item implementation."""

    def validate_image(self) -> None:
        """Validate image tensor format."""
        # check if image has shape (B, C, H, W)
        if self.image.ndim == 4 and self.image.shape[0] > 1:
            return
        msg = "Image must have shape (B, C, H, W)"
        raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label tensor format."""
        # check if label has shape (B, )
        if self.label.ndim == 2 and self.label.shape[1] == 1:
            return
        msg = "Label must have shape (B, )"
        raise ValueError(msg)

    def validate_score(self) -> None:
        """Validate score tensor format."""
        # check if score has shape (B, C)
        if self.score.ndim == 2 and self.score.shape[1] > 1:
            return
        msg = "Score must have shape (B, C)"
        raise ValueError(msg)


@dataclass
class TorchPredItemBatch(PredItemBatch[Tensor]):
    """Torch prediction data item batch implementation."""

    def validate_image(self) -> None:
        """Validate image tensor format."""
        if not (isinstance(self.images, Tensor) and self.images.ndim == 4):
            msg = f"Image must have shape (B, C, H, W), but got {self.images.shape}"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label tensor format."""
        if not (isinstance(self.labels, Tensor) and self.labels.ndim == 2):
            msg = f"Label must have shape (B, ), but got {self.labels.shape}"
            raise ValueError(msg)

    def validate_scores(self) -> None:
        """Validate prediction scores tensor format."""
        # check if scores has shape (B, C)
        if self.scores.ndim == 2 and self.scores.shape[1] > 1:
            return
        msg = f"Scores must have shape (B, C), but got {self.scores.shape}"
        raise ValueError(msg)

    def validate_feature_vector(self) -> None:
        if self.feature_vector is None:
            return
        msg = "Not implemented"
        raise NotImplementedError(msg)

    def validate_saliency_map(self) -> None:
        if self.saliency_map is None:
            return
        msg = "Not implemented"
        raise NotImplementedError(msg)
