# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification

from app.models import DatasetItemSubset

from .models import DatasetItemWithLabels, SplitRatios, SubsetAssignment


class SubsetAssigner:
    """
    Assigns dataset items to subsets ensuring balanced label representation.

    Uses iterative stratification to handle both single-label and multi-label classification problems.

    Sources:
    Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data.
    Machine Learning and Knowledge Discovery in Databases, 145-158.
    http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

    Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning
    with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017.
    http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html
    """

    def __init__(self) -> None:
        self._mlb = MultiLabelBinarizer()

    def assign(
        self, items: list[DatasetItemWithLabels], target_ratios: SplitRatios, has_all_subsets_assigned: bool
    ) -> list[SubsetAssignment]:
        """
        Assigns dataset items to subsets based on target ratios.

        Args:
            items (list[DatasetItemWithLabels]): List of dataset items to assign.
            target_ratios (SplitRatios): Desired split ratios for subsets.
            has_all_subsets_assigned (bool): Whether the project already has at least one item
                assigned to each of the TRAINING, VALIDATION, and TESTING subsets.
                - If ``False`` and fewer than 3 items are provided, a ``ValueError`` is raised
                  because there are not enough items to populate every subset.  After stratification,
                  ``_ensure_all_subsets_nonempty`` is called to guarantee no subset is left empty.
                - If ``True``, the minimum-item guard is bypassed (the caller guarantees that each
                  subset is already covered).  When fewer than 3 items are provided they are
                  assigned sequentially (TRAINING first, then VALIDATION, then TESTING).
                  ``_ensure_all_subsets_nonempty`` is *not* called, so stratification may produce
                  empty folds without raising.
        Returns:
            list[SubsetAssignment]: List of subset assignments for each item.
        """
        if len(items) < 3:
            if not has_all_subsets_assigned:
                raise ValueError(
                    "Not all subsets have items assigned, but number of unassigned dataset items is less than number "
                    "of subsets: Training, Validation and Testing. Each subset requires at least 1 item before "
                    "training can start."
                )
            # If less items are available than subsets, we cannot use stratification, so assign in order to subsets.
            assignments = []
            subsets = [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING]
            for idx in range(len(items)):
                assignments.append(SubsetAssignment(item_id=items[idx].item_id, subset=subsets[idx]))
            return assignments

        label_matrix = self._mlb.fit_transform([item.labels for item in items])

        stratifier = IterativeStratification(
            n_splits=3,
            order=1,
            sample_distribution_per_fold=target_ratios.to_list(),
        )

        X = np.arange(len(items)).reshape(-1, 1)
        y = label_matrix

        splits = list(stratifier.split(X, y))  # pyrefly: ignore[bad-argument-type]
        train_indices: list[int] = splits[0][1].tolist()
        val_indices: list[int] = splits[1][1].tolist()
        test_indices: list[int] = splits[2][1].tolist()

        indices_by_subset: dict[DatasetItemSubset, list[int]] = {
            DatasetItemSubset.TRAINING: train_indices,
            DatasetItemSubset.VALIDATION: val_indices,
            DatasetItemSubset.TESTING: test_indices,
        }

        if not has_all_subsets_assigned:
            self._ensure_all_subsets_nonempty(indices_by_subset)

        assignments = []
        for subset, indices in indices_by_subset.items():
            for idx in indices:
                assignments.append(SubsetAssignment(item_id=items[idx].item_id, subset=subset))

        return assignments

    @staticmethod
    def _ensure_all_subsets_nonempty(
        indices_by_subset: dict[DatasetItemSubset, list[int]],
    ) -> None:
        """Ensure every subset has at least one index by moving items from the largest subset."""
        empty_subsets = [s for s, idx in indices_by_subset.items() if len(idx) == 0]
        for empty_subset in empty_subsets:
            largest_subset = max(indices_by_subset, key=lambda s: len(indices_by_subset[s]))
            moved = indices_by_subset[largest_subset].pop()
            indices_by_subset[empty_subset].append(moved)
