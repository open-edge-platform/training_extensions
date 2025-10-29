# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.models import DatasetItemSubset
from app.services.training.subset_assignment import DatasetItemWithLabels, SplitRatios, SubsetAssigner


class TestSubsetAssigner:
    """Test cases for SubsetAssigner class"""

    @pytest.fixture
    def fxt_assigner(self):
        """Create a SubsetAssigner instance"""
        return SubsetAssigner()

    @pytest.fixture
    def fxt_default_ratios(self):
        """Create default split ratios"""
        return SplitRatios(train=0.7, val=0.2, test=0.1)

    def test_assign_empty_list(self, fxt_assigner, fxt_default_ratios):
        """Test that assigning empty list returns empty list"""
        result = fxt_assigner.assign([], fxt_default_ratios)
        assert result == []

    def test_assign_returns_all_items(self, fxt_assigner, fxt_default_ratios):
        """Test that all input items are assigned to a subset"""
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(100)]

        result = fxt_assigner.assign(items, fxt_default_ratios)

        assert len(result) == len(items)

        # Check all item IDs are present
        result_item_ids = {assignment.item_id for assignment in result}
        input_item_ids = {item.item_id for item in items}
        assert result_item_ids == input_item_ids

    def test_assign_maintains_label_distribution(self, fxt_assigner, fxt_default_ratios):
        """Test that label distribution is maintained across subsets (stratification)"""
        label_a = uuid4()
        label_b = uuid4()

        items = []
        # 60% label A, 40% label B
        for _ in range(60):
            items.append(DatasetItemWithLabels(item_id=uuid4(), labels={label_a}))
        for _ in range(40):
            items.append(DatasetItemWithLabels(item_id=uuid4(), labels={label_b}))

        result = fxt_assigner.assign(items, fxt_default_ratios)

        item_labels_map = {item.item_id: item.labels for item in items}

        # Check label distribution in each subset
        for subset in [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING]:
            subset_assignments = [a for a in result if a.subset == subset]

            label_a_count = sum(1 for a in subset_assignments if label_a in item_labels_map[a.item_id])
            label_b_count = sum(1 for a in subset_assignments if label_b in item_labels_map[a.item_id])

            total_in_subset = len(subset_assignments)
            if total_in_subset > 0:
                # Label A should be ~60% and label B ~40% in each subset
                assert label_a_count / total_in_subset == 0.6
                assert label_b_count / total_in_subset == 0.4
