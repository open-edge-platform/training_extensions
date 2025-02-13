"""NumPy-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import DataItem, DataItemBatch, PredItem, PredItemBatch


@dataclass
class NumpyDataItem(DataItem[NDArray]):
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
            msg = "Image must be a float32 array with shape (H, W, 3)"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label array format.

        Raises:
            ValueError: If label is not a scalar or 1D int64 array
        """
        if not isinstance(self.label, np.ndarray):
            msg = "Label must be a numpy.ndarray"
            raise TypeError(msg)

        if not (
            (self.label.ndim == 0 and self.label.dtype == np.int64)  # Multi-class
            or (self.label.ndim == 1 and self.label.dtype == np.int64)  # Multi-label/hierarchical
        ):
            msg = "Label must be a scalar or 1D int64 array"
            raise ValueError(msg)


@dataclass
class NumpyDataItemBatch(DataItemBatch[NDArray]):
    """NumPy data item batch implementation."""

    def validate_image(self) -> None:
        """Validate image array format."""
        if not (isinstance(self.images, np.ndarray) and self.images.ndim == 4):
            msg = f"Images must have shape (B, H, W, C), but got {self.images.shape}"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label array format."""
        if not (isinstance(self.labels, np.ndarray) and self.labels.ndim == 2):
            msg = f"Labels must have shape (B, C), but got {self.labels.shape}"
            raise ValueError(msg)

    def validate_masks(self) -> None:
        """Validate masks array format."""
        if self.masks is None:
            return
        if not (isinstance(self.masks, np.ndarray) and self.masks.ndim == 4):
            msg = f"Masks must have shape (B, H, W, C), but got {self.masks.shape}"
            raise ValueError(msg)

    def validate_boxes(self) -> None:
        """Validate boxes array format."""
        if self.boxes is None:
            return
        if not (isinstance(self.boxes, np.ndarray) and self.boxes.ndim == 3):
            msg = f"Boxes must have shape (B, N, 4), but got {self.boxes.shape}"
            raise ValueError(msg)

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
class NumpyPredItem(PredItem[NDArray]):
    """NumPy prediction item implementation."""

    def validate_image(self) -> None:
        """Validate image array format."""
        if not (isinstance(self.image, np.ndarray) and self.image.ndim == 4):
            msg = f"Image must have shape (B, H, W, C), but got {self.image.shape}"
            raise ValueError(msg)

    def validate_label(self) -> None:
        """Validate label array format."""
        if not (isinstance(self.label, np.ndarray) and self.label.ndim == 2):
            msg = f"Label must have shape (B, C), but got {self.label.shape}"
            raise ValueError(msg)

    def validate_score(self) -> None:
        """Validate score array format."""
        if not (isinstance(self.score, np.ndarray) and self.score.ndim == 2):
            msg = f"Score must have shape (B, C), but got {self.score.shape}"
            raise ValueError(msg)


@dataclass
class NumpyPredItemBatch(PredItemBatch[NDArray]):
    """NumPy prediction batch implementation."""

    def validate_scores(self) -> None:
        """Validate scores array format."""
        if not (isinstance(self.scores, np.ndarray) and self.scores.ndim == 2):
            msg = f"Scores must have shape (B, C), but got {self.scores.shape}"
            raise ValueError(msg)

    def validate_saliency_map(self) -> None:
        """Validate saliency map array format."""
        if self.saliency_map is None:
            return
        if not (isinstance(self.saliency_map, np.ndarray) and self.saliency_map.ndim == 4):
            msg = f"Saliency map must have shape (B, H, W, C), but got {self.saliency_map.shape}"
            raise ValueError(msg)

    def validate_feature_vector(self) -> None:
        """Validate feature vector array format."""
        if self.feature_vector is None:
            return
        if not (isinstance(self.feature_vector, np.ndarray) and self.feature_vector.ndim == 2):
            msg = f"Feature vector must have shape (B, D), but got {self.feature_vector.shape}"
            raise ValueError(msg)
