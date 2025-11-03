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

    def assign(self, items: list[DatasetItemWithLabels], target_ratios: SplitRatios) -> list[SubsetAssignment]:
        """
        Assigns dataset items to subsets based on target ratios.

        Args:
            items (list[DatasetItemWithLabels]): List of dataset items to assign.
            target_ratios (SplitRatios): Desired split ratios for subsets.
        Returns:
            list[SubsetAssignment]: List of subset assignments for each item.
        """
        if not items:
            return []

        label_matrix = self._mlb.fit_transform([item.labels for item in items])

        stratifier = IterativeStratification(
            n_splits=3,
            order=1,
            sample_distribution_per_fold=target_ratios.to_list(),
        )

        X = np.arange(len(items)).reshape(-1, 1)
        y = label_matrix

        splits = list(stratifier.split(X, y))
        train_indices = splits[0][1]
        val_indices = splits[1][1]
        test_indices = splits[2][1]

        indices_by_subset: dict[DatasetItemSubset, list[int]] = {
            DatasetItemSubset.TRAINING: train_indices,
            DatasetItemSubset.VALIDATION: val_indices,
            DatasetItemSubset.TESTING: test_indices,
        }

        assignments = []
        for subset, indices in indices_by_subset.items():
            for idx in indices:
                assignments.append(SubsetAssignment(item_id=items[idx].item_id, subset=subset))

        return assignments
