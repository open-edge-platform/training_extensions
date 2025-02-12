"""Dataclasses for Torch data items."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import DataItem, DataItemBatch, PredDataItem


@dataclass
class TorchDataItem(DataItem):
    """Torch data item."""

    image: Tensor
    label: Tensor


@dataclass
class TorchDataItemBatch(DataItemBatch):
    """Torch data item batch."""

    images: Tensor
    labels: Tensor

    @classmethod
    def collate_fn(cls, items: list[TorchDataItem]) -> TorchDataItemBatch:
        """Collate function for TorchDataItemBatch."""
        return TorchDataItemBatch(
            images=torch.stack([item.image for item in items]),
            labels=torch.stack([item.label for item in items]),
        )


@dataclass
class TorchPredDataItem(PredDataItem):
    """Torch pred data item."""

    images: Tensor
    labels: Tensor
    scores: Tensor
