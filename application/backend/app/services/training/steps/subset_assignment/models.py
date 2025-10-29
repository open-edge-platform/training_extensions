# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from uuid import UUID

from app.models import DatasetItemSubset


@dataclass(frozen=True)
class SplitRatios:
    """Value object representing split ratios for subset assignment"""

    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if not (0.99 <= total <= 1.01):  # Allow for floating point precision issues
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def to_fold_sizes(self, total_items: int) -> dict[DatasetItemSubset, int]:
        """Convert split ratios to absolute fold sizes based on total items"""
        train_size = int(total_items * self.train)
        val_size = int(total_items * self.val)
        test_size = total_items - (train_size + val_size)
        return {
            DatasetItemSubset.TRAINING: train_size,
            DatasetItemSubset.VALIDATION: val_size,
            DatasetItemSubset.TESTING: test_size,
        }

    def to_list(self) -> list[float]:
        """Convert split ratios to a list"""
        return [self.train, self.val, self.test]


@dataclass(frozen=True)
class SubsetAssignment:
    """Value object representing assignment of a dataset item to a subset"""

    item_id: UUID
    subset: DatasetItemSubset


@dataclass(frozen=True)
class DatasetItemWithLabels:
    """Representation of a dataset item with its associated labels for assignment processing."""

    item_id: UUID
    labels: set[UUID]
