# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune data entities."""

from .base import ImageInfo, ImageType
from .sample import (
    Prediction,
    PredictionBatch,
    BaseSample,
    SampleBatch,
)
from .tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileSegDataEntity,
)

__all__ = [
    "ImageInfo",
    "ImageType",
    "Prediction",
    "PredictionBatch",
    "BaseSample",
    "SampleBatch",
    "TileBatchDetDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchSegDataEntity",
    "TileDetDataEntity",
    "TileSegDataEntity",
]
