# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX data entities."""

from .base import ImageInfo, ImageType, Points
from .sample import (
    OTXPrediction,
    OTXPredictionBatch,
    OTXSample,
    OTXSampleBatch,
    collate_fn,
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
    "OTXPrediction",
    "OTXPredictionBatch",
    "OTXSample",
    "OTXSampleBatch",
    "Points",
    "TileBatchDetDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchSegDataEntity",
    "TileDetDataEntity",
    "TileSegDataEntity",
    "collate_fn",
]
