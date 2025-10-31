# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.models import DatasetItemSubset

from .models import SplitRatios


class SubsetDistribution:
    """Represents current distribution of items across subsets."""

    def __init__(self, counts: dict[DatasetItemSubset, int]):
        self._counts = counts.copy()

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def get_count(self, subset: DatasetItemSubset) -> int:
        return self._counts.get(subset, 0)

    def compute_adjusted_ratios(self, target_ratios: SplitRatios, unassigned_count: int) -> SplitRatios:
        """
        Compute adjusted ratios for unassigned items to reach target distribution.

        This method calculates how unassigned items should be distributed to achieve the overall target ratios.
        """
        if unassigned_count < 0:
            raise ValueError(f"Parameter 'unassigned_count' cannot be negative (got {unassigned_count})")
        if unassigned_count == 0:
            return target_ratios

        target_counts = target_ratios.to_fold_sizes(self.total + unassigned_count)

        # Calculate how many more items are needed in each subset
        needed = {
            subset: max(0, target_counts[subset] - self.get_count(subset))
            for subset in DatasetItemSubset
            if subset != DatasetItemSubset.UNASSIGNED
        }

        # Normalize
        total_needed = sum(needed.values())
        adjusted = {k: v / total_needed for k, v in needed.items()}

        return SplitRatios(
            train=adjusted[DatasetItemSubset.TRAINING],
            val=adjusted[DatasetItemSubset.VALIDATION],
            test=adjusted[DatasetItemSubset.TESTING],
        )

    def __repr__(self) -> str:
        return f"SubsetDistribution(counts={self._counts})"
