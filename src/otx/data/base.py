"""Dataclasses for dataset items."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class DataItem(ABC):
    """Data item."""

    # TODO(ashwinvaidya17): This is temporary and needs to be replaced with the actual media type
    # media: Any
    label: Any
    # mask: Any
    # bboxes: Any


@dataclass
class DataItemBatch(ABC):
    """Data item batch."""
    # data_items: list[DataItem]


@dataclass
class PredDataItem(ABC):
    """Pred data item."""
    # mask: Any
    # bboxes: Any
    # label: Any
    # score: Any
