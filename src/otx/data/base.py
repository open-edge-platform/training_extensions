"""Base classes for dataset items."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

TENSOR_TYPE = TypeVar("TENSOR_TYPE")

@dataclass
class BaseItem(Generic[TENSOR_TYPE], ABC):
    """Abstract base class for all data items."""

    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        for method_name in dir(self):
            if method_name.startswith('validate_'):
                method = getattr(self, method_name)
                if callable(method):
                    method()

@dataclass
class DataItem(BaseItem[TENSOR_TYPE], ABC):
    """Abstract base class for data items.

    Args:
        image: Image data
        label: Label data
    """
    image: TENSOR_TYPE
    label: TENSOR_TYPE
    # mask: Any
    # bboxes: Any

    @abstractmethod
    def validate_image(self) -> None:
        """Validate image data format and type."""

    @abstractmethod
    def validate_label(self) -> None:
        """Validate label data format and type."""

@dataclass
class DataItemBatch(BaseItem[TENSOR_TYPE], ABC):
    """Abstract base class for batched data items.

    Args:
        images: Batch of image data of type T
        labels: Batch of label data of type L
    """

    images: TENSOR_TYPE
    labels: TENSOR_TYPE
    masks: TENSOR_TYPE | None = None
    boxes: TENSOR_TYPE | None = None

    @abstractmethod
    def validate_image(self) -> None:
        """Validate image data format and type."""

    @abstractmethod
    def validate_label(self) -> None:
        """Validate label data format and type."""

    @abstractmethod
    def validate_masks(self) -> None:
        """Validate masks data format and type."""

    @abstractmethod
    def validate_boxes(self) -> None:
        """Validate boxes data format and type."""

@dataclass
class PredItem(DataItem[TENSOR_TYPE], ABC):
    """Abstract base class for prediction data items.

    Args:
        image: Image data
        label: Predicted label data
        scores: Confidence scores for predictions
    """

    score: TENSOR_TYPE

    @abstractmethod
    def validate_score(self) -> None:
        """Validate prediction score format and type."""

    @abstractmethod
    def validate_feature_vector(self) -> None:
        """Validate feature vector format and type."""

    @abstractmethod
    def validate_saliency_map(self) -> None:
        """Validate saliency map format and type."""

@dataclass
class PredItemBatch(DataItemBatch[TENSOR_TYPE], ABC):
    """Abstract base class for batched prediction data items.

    Args:
        images: Batch of image data
        labels: Batch of label data
    """

    scores: TENSOR_TYPE | None = None
    saliency_map: TENSOR_TYPE | None = None
    feature_vector: TENSOR_TYPE | None = None

    @abstractmethod
    def validate_scores(self) -> None:
        """Validate prediction scores format and type."""

    @abstractmethod
    def validate_saliency_map(self) -> None:
        """Validate saliency map format and type."""

    @abstractmethod
    def validate_feature_vector(self) -> None:
        """Validate feature vector format and type."""
