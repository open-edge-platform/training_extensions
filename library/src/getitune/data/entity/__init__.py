# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for Geti Tune data entities."""

from .base import ImageInfo, ImageType
from .sample import (
    OTXPrediction,
    OTXPredictionBatch,
    OTXSample,
    OTXSampleBatch,
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
    "TileBatchDetDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchSegDataEntity",
    "TileDetDataEntity",
    "TileSegDataEntity",
]
