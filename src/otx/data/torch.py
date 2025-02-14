"""Torch-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import NoneType
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_type_hints

import numpy as np
import torch
from torch import Tensor

from .validations import (
    validate_boxes_batch,
    validate_feature_vector_and_batch,
    validate_image,
    validate_image_batch,
    validate_label,
    validate_label_batch,
    validate_masks_batch,
    validate_saliency_map,
    validate_saliency_map_batch,
    validate_scores,
)


@dataclass
class TorchDataItem:
    """Torch data item implementation."""

    image: torch.Tensor
    label: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.image = validate_image(self.image)
        self.label = validate_label(self.label)

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


@dataclass
class TorchDataItemBatch:
    """Torch data item batch implementation."""

    images: torch.Tensor
    labels: torch.Tensor | None
    masks: torch.Tensor | None = None
    boxes: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.images = validate_image_batch(self.images)
        self.labels = validate_label_batch(self.labels)
        self.masks = validate_masks_batch(self.masks) if self.masks is not None else None
        self.boxes = validate_boxes_batch(self.boxes) if self.boxes is not None else None


@dataclass
class TorchPredItem:
    """Torch prediction data item implementation."""

    image: torch.Tensor
    label: torch.Tensor | None
    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.image = validate_image(self.image)
        self.label = validate_label(self.label)
        self.scores = validate_scores(self.scores) if self.scores is not None else None
        self.feature_vector = (
            validate_feature_vector_and_batch(self.feature_vector) if self.feature_vector is not None else None
        )
        self.saliency_map = validate_saliency_map(self.saliency_map) if self.saliency_map is not None else None


@dataclass
class TorchPredItemBatch:
    """Torch prediction data item batch implementation."""

    images: torch.Tensor
    labels: torch.Tensor | None
    scores: torch.Tensor | None = None
    feature_vectors: torch.Tensor | None = None
    saliency_maps: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    boxes: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.images = validate_image_batch(self.images)
        self.labels = validate_label_batch(self.labels)
        self.scores = validate_scores(self.scores) if self.scores is not None else None
        self.feature_vectors = (
            validate_feature_vector_and_batch(self.feature_vectors) if self.feature_vectors is not None else None
        )
        self.saliency_maps = validate_saliency_map_batch(self.saliency_maps) if self.saliency_maps is not None else None
        self.masks = validate_masks_batch(self.masks) if self.masks is not None else None
        self.boxes = validate_boxes_batch(self.boxes) if self.boxes is not None else None
