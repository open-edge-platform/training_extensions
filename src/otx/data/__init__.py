"""Dataclasses for data entities."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .numpy import (
    NumpyDataBatch,
    NumpyDataItem,
)
from .torch import (
    TorchDataBatch,
    TorchDataItem,
    TorchPredBatch,
    TorchPredItem,
)

__all__ = [
    "TorchDataBatch",
    "TorchDataItem",
    "TorchPredBatch",
    "TorchPredItem",
    "NumpyDataBatch",
    "NumpyDataItem",
]
