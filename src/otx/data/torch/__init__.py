"""Torch entities."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import TorchDataBatch, TorchDataItem, TorchPredBatch, TorchPredItem
from .tile import TorchTileDataBatch, TorchTileDataItem

__all__ = [
    "TorchDataBatch",
    "TorchDataItem",
    "TorchPredBatch",
    "TorchPredItem",
    "TorchTileDataBatch",
    "TorchTileDataItem",
]
