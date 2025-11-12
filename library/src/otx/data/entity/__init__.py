# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX data entities."""

from .base import ImageInfo, ImageType, Points
from .tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileSegDataEntity,
)
from .torch import OTXDataBatch, OTXDataItem, OTXPredBatch, OTXPredItem

__all__ = [
    "ImageInfo",
    "ImageType",
    "OTXDataBatch",
    "OTXDataItem",
    "OTXPredBatch",
    "OTXPredItem",
    "Points",
    "TileBatchDetDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchSegDataEntity",
    "TileDetDataEntity",
    "TileSegDataEntity",
]
