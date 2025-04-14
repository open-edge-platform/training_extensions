"""Numpy-specific data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from datumaro import Polygon

    from otx.core.data.entity.base import ImageInfo


@dataclass
class NumpyDataItem:
    """Data item for Numpy data."""

    image: np.ndarray | None = None
    label: np.ndarray | None = None
    masks: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    polygons: list[Polygon] | None = None
    img_info: ImageInfo | None = None

    @staticmethod
    def collate_fn(items: list[NumpyDataItem]) -> NumpyDataBatch:
        """Collate NumpyDataItems into a batch.

        Args:
            items: List of NumpyDataItems to batch
        Returns:
            Batched NumpyDataItems with stacked tensors
        """
        # Implement the collate function for NumpyDataItem
        if all(item.image.shape == items[0].image.shape for item in items):
            images = np.stack([item.image for item in items])
        else:
            # we need this only in case of OV inference, where no resize
            images = [item.image for item in items]

        return NumpyDataBatch(
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


@dataclass
class NumpyDataBatch:
    batch_size: int
    images: np.ndarray | list[np.ndarray] | None = None
    labels: list[np.ndarray] | None = None
    masks: list[np.ndarray] | None = None
    bboxes: list[np.ndarray] | None = None
    keypoints: list[np.ndarray] | None = None
    polygons: list[list[Polygon]] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None


@dataclass
class NumpyPredBatch(NumpyDataBatch):
    """Numpy data item batch implementation."""

    scores: list[np.ndarray] | None = None
    feature_vector: list[np.ndarray] | None = None
    saliency_map: list[np.ndarray] | None = None
