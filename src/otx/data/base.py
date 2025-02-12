"""Base classes for dataset items."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar('T')  # Type for image/tensor data
L = TypeVar('L')  # Type for label data

@dataclass
class DataItem(Generic[T, L], ABC):
    """Abstract base class for data items.
    
    Args:
        image: Image data of type T
        label: Label data of type L
    """

    image: T
    label: L
    # mask: Any
    # bboxes: Any

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self.validate_image()
        self.validate_label()

    @abstractmethod
    def validate_image(self) -> None:
        """Validate image data format and type."""

    @abstractmethod
    def validate_label(self) -> None:
        """Validate label data format and type."""


@dataclass
class DataItemBatch(Generic[T, L], ABC):
    """Abstract base class for batched data items.
    
    Args:
        images: Batch of image data of type T
        labels: Batch of label data of type L
    """

    images: T
    labels: L

    def __post_init__(self) -> None:
        """Validate all data after initialization."""
        self.validate_image()
        self.validate_label()

    @abstractmethod
    def validate_image(self) -> None:
        """Validate image data format and type."""

    @abstractmethod
    def validate_label(self) -> None:
        """Validate label data format and type."""


@dataclass
class PredDataItem(DataItem[T, L], ABC):
    """Abstract base class for prediction data items.
    
    Args:
        image: Image data of type T
        label: Predicted label data of type L
        scores: Confidence scores for predictions of type T
    """

    score: T

    @abstractmethod
    def validate_score(self) -> None:
        """Validate prediction score format and type."""

    def __post_init__(self) -> None:
        """Validate all data after initialization."""
        super().__post_init__()
        self.validate_score()

@dataclass
class PredDataItemBatch(DataItemBatch[T, L], ABC):
    """Abstract base class for batched prediction data items.
    
    Args:
        images: Batch of image data of type T
        labels: Batch of label data of type L
    """
    scores: T

    def __post_init__(self) -> None:
        """Validate all data after initialization."""
        super().__post_init__()
        self.validate_scores()

    @abstractmethod
    def validate_scores(self) -> None:
        """Validate prediction scores format and type."""
        

