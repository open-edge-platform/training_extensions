"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from .base import DataItem, DataItemBatch, PredItem, PredItemBatch


class TorchValidationMixin:
    """Mixin class providing common tensor validation methods."""

    def _validate_tensor(
        self,
        tensor: Any,
        expected_ndim: int,
        expected_dtype: torch.dtype,
        name: str,
        optional: bool = False,
        **dim_constraints: int,
    ) -> None:
        """Validate tensor properties.

        Args:
            tensor: Tensor to validate
            expected_ndim: Expected number of dimensions
            expected_dtype: Expected dtype
            name: Name of tensor for error messages
            optional: Whether tensor can be None
            **dim_constraints: Expected size for specific dimensions (e.g. dim0=3)
        """
        if optional and tensor is None:
            return

        if not isinstance(tensor, Tensor):
            msg = f"{name} must be a torch.Tensor"
            raise TypeError(msg)

        if tensor.ndim != expected_ndim:
            msg = f"{name} must have {expected_ndim} dimensions, got {tensor.ndim}"
            raise ValueError(msg)

        if tensor.dtype != expected_dtype:
            msg = f"{name} must have dtype {expected_dtype}, got {tensor.dtype}"
            raise ValueError(msg)

        for dim, size in dim_constraints.items():
            if tensor.shape[int(dim[-1])] != size:
                msg = f"{name} must have size {size} in dimension {dim}, got {tensor.shape[int(dim[-1])]}"
                raise ValueError(msg)


@dataclass
class TorchDataItem(DataItem[Tensor], TorchValidationMixin):
    """Torch data item implementation.

    Args:
        image: Image tensor of shape (C, H, W) and type float32
        label: Label tensor, either scalar for multi-class or 1D for multi-label
    """

    # Class constants for validation
    IMAGE_CHANNELS: ClassVar[int] = 3

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
            labels=torch.vstack([item.label for item in items]),
        )

    def validate_image(self) -> None:
        """Validate image tensor format."""
        self._validate_tensor(
            self.image,
            expected_ndim=3,
            expected_dtype=torch.float32,
            name="Image",
            dim0=self.IMAGE_CHANNELS,
        )

    def validate_label(self) -> None:
        """Validate label tensor format."""
        if self.label.ndim == 0:  # Multi-class
            self._validate_tensor(
                self.label,
                expected_ndim=0,
                expected_dtype=torch.long,
                name="Label",
            )
        else:  # Multi-label/hierarchical
            self._validate_tensor(
                self.label,
                expected_ndim=1,
                expected_dtype=torch.long,
                name="Label",
            )


@dataclass
class TorchDataItemBatch(DataItemBatch[Tensor], TorchValidationMixin):
    """Torch data item batch implementation."""

    def validate_image(self) -> None:
        """Validate image tensor format."""
        self._validate_tensor(
            self.images,
            expected_ndim=4,
            expected_dtype=torch.float32,
            name="Images",
            dim1=TorchDataItem.IMAGE_CHANNELS,
        )

    def validate_label(self) -> None:
        """Validate label tensor format."""
        self._validate_tensor(
            self.labels,
            expected_ndim=2,
            expected_dtype=torch.long,
            name="Labels",
        )

    def validate_masks(self) -> None:
        """Validate masks tensor format."""
        self._validate_tensor(
            self.masks,
            expected_ndim=4,
            expected_dtype=torch.bool,
            name="Masks",
            optional=True,
        )

    def validate_boxes(self) -> None:
        """Validate boxes tensor format."""
        self._validate_tensor(
            self.boxes,
            expected_ndim=3,
            expected_dtype=torch.float32,
            name="Boxes",
            optional=True,
            dim2=4,  # boxes should have 4 coordinates
        )


@dataclass
class TorchPredItem(PredItem[Tensor], TorchValidationMixin):
    """Torch prediction data item implementation."""

    def validate_image(self) -> None:
        """Validate image tensor format."""
        self._validate_tensor(
            self.image,
            expected_ndim=4,
            expected_dtype=torch.float32,
            name="Image",
            dim1=TorchDataItem.IMAGE_CHANNELS,
        )

    def validate_label(self) -> None:
        """Validate label tensor format."""
        self._validate_tensor(
            self.label,
            expected_ndim=2,
            expected_dtype=torch.long,
            name="Label",
        )

    def validate_score(self) -> None:
        """Validate score tensor format."""
        self._validate_tensor(
            self.score,
            expected_ndim=2,
            expected_dtype=torch.float32,
            name="Score",
        )

    def validate_feature_vector(self) -> None:
        """Validate feature vector format."""
        self._validate_tensor(
            self.feature_vector,
            expected_ndim=2,
            expected_dtype=torch.float32,
            name="Feature vector",
            optional=True,
        )

    def validate_saliency_map(self) -> None:
        """Validate saliency map format."""
        self._validate_tensor(
            self.saliency_map,
            expected_ndim=4,
            expected_dtype=torch.float32,
            name="Saliency map",
            optional=True,
        )


@dataclass
class TorchPredItemBatch(PredItemBatch[Tensor], TorchValidationMixin):
    """Torch prediction data item batch implementation."""

    def validate_scores(self) -> None:
        """Validate scores tensor format."""
        self._validate_tensor(
            self.scores,
            expected_ndim=2,
            expected_dtype=torch.float32,
            name="Scores",
            optional=True,
        )

    def validate_saliency_map(self) -> None:
        """Validate saliency map tensor format."""
        self._validate_tensor(
            self.saliency_map,
            expected_ndim=4,
            expected_dtype=torch.float32,
            name="Saliency map",
            optional=True,
        )

    def validate_feature_vector(self) -> None:
        """Validate feature vector tensor format."""
        self._validate_tensor(
            self.feature_vector,
            expected_ndim=2,
            expected_dtype=torch.float32,
            name="Feature vector",
            optional=True,
        )

    def validate_boxes(self) -> None:
        if self.boxes is None:
            return
        msg = "Not implemented"
        raise NotImplementedError(msg)
    
    def validate_masks(self) -> None:
        if self.masks is None:
            return
        msg = "Not implemented"
        raise NotImplementedError(msg)
