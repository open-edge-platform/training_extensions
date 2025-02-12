"""NumPy-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import DataItem, DataItemBatch, PredItem


@dataclass
class NumpyDataItem(DataItem[NDArray, NDArray]):
    """NumPy data item implementation.

    Args:
        image: Image array of shape (H, W, C) and type float32
        label: Label array, either scalar for multi-class or 1D for multi-label
    """

    def validate_image(self) -> None:
        """Validate image array format.

        Raises:
            ValueError: If image is not a 3D float32 array with shape (H, W, 3)
        """
        if not (
            isinstance(self.image, np.ndarray)
            and self.image.ndim == 3
            and self.image.shape[-1] == 3
            and self.image.dtype == np.float32
        ):
            raise ValueError("Image must be a float32 array with shape (H, W, 3)")

    def validate_label(self) -> None:
        """Validate label array format.

        Raises:
            ValueError: If label is not a scalar or 1D int64 array
        """
        if not isinstance(self.label, np.ndarray):
            raise ValueError("Label must be a numpy.ndarray")

        if not (
            (self.label.ndim == 0 and self.label.dtype == np.int64)  # Multi-class
            or (self.label.ndim == 1 and self.label.dtype == np.int64)  # Multi-label/hierarchical
        ):
            raise ValueError("Label must be a scalar or 1D int64 array")


@dataclass
class NumpyDataItemBatch(DataItemBatch[NDArray, NDArray]):
    """NumPy data item batch implementation."""

    @classmethod
    def collate_fn(cls, items: list[NumpyDataItem]) -> "NumpyDataItemBatch":
        """Collate NumpyDataItems into a batch.

        Args:
            items: List of NumpyDataItems to batch

        Returns:
            Batched NumpyDataItems with stacked arrays
        """
        return cls(
            images=np.stack([item.image for item in items]),
            labels=np.stack([item.label for item in items]),
        )


@dataclass
class NumpyPredDataItem(PredItem[NDArray, NDArray]):
    """NumPy prediction data item implementation."""

    def validate_image(self) -> None:
        """Validate image array format."""
        raise NotImplementedError("NumpyPredDataItem does not implement validate_image")

    def validate_label(self) -> None:
        """Validate label array format."""
        raise NotImplementedError("NumpyPredDataItem does not implement validate_label")

    def validate_scores(self) -> None:
        """Validate prediction scores array format."""
        raise NotImplementedError("NumpyPredDataItem does not implement validate_scores")
