# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX data entities."""

from .tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileSegDataEntity,
)
from .torch import OTXDataBatch, OTXDataItem, OTXPredBatch, OTXPredItem
from .base import ImageInfo, ImageType

__all__ = [
    "OTXDataItem",
    "OTXDataBatch",
    "OTXPredBatch",
    "OTXPredItem",
    "TileDetDataEntity",
    "TileSegDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchDetDataEntity",
    "TileBatchSegDataEntity",
    "TileBatchInstSegDataEntity",
    "ImageType",
    "ImageInfo",
]
