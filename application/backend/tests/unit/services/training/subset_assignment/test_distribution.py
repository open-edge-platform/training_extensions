# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import re

import pytest

from app.models import DatasetItemSubset
from app.services.training.subset_assignment import SplitRatios, SubsetDistribution


class TestSubsetDistribution:
    """Test cases for SubsetDistribution class"""

    def test_create_distribution_empty(self):
        """Test creation of empty distribution"""
        distribution = SubsetDistribution(counts={})
        assert distribution.total == 0

    def test_create_distribution_with_counts(self):
        """Test creation of distribution with initial counts"""
        counts = {
            DatasetItemSubset.TRAINING: 70,
            DatasetItemSubset.VALIDATION: 20,
            DatasetItemSubset.TESTING: 10,
        }
        distribution = SubsetDistribution(counts=counts)

        assert distribution.get_count(DatasetItemSubset.TRAINING) == 70
        assert distribution.get_count(DatasetItemSubset.VALIDATION) == 20
        assert distribution.get_count(DatasetItemSubset.TESTING) == 10
        assert distribution.total == 100

    def test_get_count_existing_subset(self):
        """Test getting count for an existing subset"""
        counts = {DatasetItemSubset.TRAINING: 42}
        distribution = SubsetDistribution(counts=counts)

        assert distribution.get_count(DatasetItemSubset.TRAINING) == 42

    def test_get_count_missing_subset(self):
        """Test getting count for a subset not in distribution returns 0"""
        counts = {DatasetItemSubset.TRAINING: 42}
        distribution = SubsetDistribution(counts=counts)

        assert distribution.get_count(DatasetItemSubset.VALIDATION) == 0
        assert distribution.get_count(DatasetItemSubset.TESTING) == 0

    def test_counts_are_copied(self):
        """Test that internal counts are a copy, not a reference"""
        original_counts = {DatasetItemSubset.TRAINING: 50}
        distribution = SubsetDistribution(counts=original_counts)

        # Modify original
        original_counts[DatasetItemSubset.TRAINING] = 100

        # Distribution should still have original value
        assert distribution.get_count(DatasetItemSubset.TRAINING) == 50


class TestComputeAdjustedRatios:
    """Test cases for compute_adjusted_ratios method"""

    def test_no_unassigned_items(self):
        """Test that target ratios are returned when there are no unassigned items"""
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 70,
                DatasetItemSubset.VALIDATION: 20,
                DatasetItemSubset.TESTING: 10,
            }
        )
        target_ratios = SplitRatios(train=0.8, val=0.1, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=0)

        assert result == target_ratios

    def test_empty_distribution_with_unassigned(self):
        """Test adjusted ratios when starting from empty distribution"""
        distribution = SubsetDistribution(counts={})
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=100)

        # With empty distribution, adjusted ratios should match target ratios
        assert result.train == 0.7
        assert result.val == 0.2
        assert result.test == 0.1

    def test_partial_distribution_needs_rebalancing(self):
        """Test adjusted ratios when existing distribution needs rebalancing"""
        # Current distribution: 40 train, 10 val, 0 test = 50 total
        # Target: 70% train, 20% val, 10% test
        # With 50 unassigned, total will be 100
        # Target counts: 70 train, 20 val, 10 test
        # Needed: 30 train, 10 val, 10 test
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 40,
                DatasetItemSubset.VALIDATION: 10,
                DatasetItemSubset.TESTING: 0,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=50)

        # 30 train, 10 val, 10 test needed out of 50 unassigned
        assert result.train == pytest.approx(30 / 50)  # 0.6
        assert result.val == pytest.approx(10 / 50)  # 0.2
        assert result.test == pytest.approx(10 / 50)  # 0.2

    def test_already_exceeded_target_in_some_subsets(self):
        """Test when some subsets already exceed their target"""
        # Current: 80 train, 10 val, 10 test = 100 total
        # Target: 70% train, 20% val, 10% test for 150 total
        # Target counts: 105 train, 30 val, 15 test
        # Needed: 25 train, 20 val, 5 test (training already has extra)
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 80,
                DatasetItemSubset.VALIDATION: 10,
                DatasetItemSubset.TESTING: 10,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=50)

        # Training needs 25, val needs 20, test needs 5 out of 50
        assert result.train == pytest.approx(25 / 50)  # 0.5
        assert result.val == pytest.approx(20 / 50)  # 0.4
        assert result.test == pytest.approx(5 / 50)  # 0.1

    def test_all_subsets_meet_target(self):
        """Test when all subsets already meet their target"""
        # Current distribution already satisfies target
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 70,
                DatasetItemSubset.VALIDATION: 20,
                DatasetItemSubset.TESTING: 10,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=10)

        assert result == target_ratios

    def test_one_subset_needs_all_unassigned(self):
        """Test when only one subset needs items"""
        # Current: 70 train, 20 val, 0 test = 90 total
        # Target: 70% train, 20% val, 10% test for 100 total
        # Target counts: 70 train, 20 val, 10 test
        # Needed: 0 train, 0 val, 10 test
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 70,
                DatasetItemSubset.VALIDATION: 20,
                DatasetItemSubset.TESTING: 0,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=10)

        # All 10 unassigned should go to test
        assert result.train == 0.0
        assert result.val == 0.0
        assert result.test == 1.0

    def test_small_numbers(self):
        """Test with small numbers to verify integer arithmetic"""
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 5,
                DatasetItemSubset.VALIDATION: 3,
                DatasetItemSubset.TESTING: 2,
            }
        )
        target_ratios = SplitRatios(train=0.6, val=0.3, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=10)

        # Total will be 20, target: 12 train, 6 val, 2 test
        # Needed: 7 train, 3 val, 0 test out of 10 unassigned
        assert result.train == pytest.approx(7 / 10)
        assert result.val == pytest.approx(3 / 10)
        assert result.test == pytest.approx(0.0)

    def test_large_dataset(self):
        """Test with large dataset numbers"""
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 7000,
                DatasetItemSubset.VALIDATION: 2000,
                DatasetItemSubset.TESTING: 1000,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=5000)

        # Total will be 15000, target: 10500 train, 3000 val, 1500 test
        # Needed: 3500 train, 1000 val, 500 test out of 5000
        assert result.train == pytest.approx(3500 / 5000)
        assert result.val == pytest.approx(1000 / 5000)
        assert result.test == pytest.approx(500 / 5000)

    def test_extreme_imbalance(self):
        """Test with extreme imbalance in current distribution"""
        # All items currently in training
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 100,
                DatasetItemSubset.VALIDATION: 0,
                DatasetItemSubset.TESTING: 0,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=100)

        # Total will be 200, target: 140 train, 40 val, 20 test
        # Needed: 40 train, 40 val, 20 test out of 100
        assert result.train == pytest.approx(40 / 100)
        assert result.val == pytest.approx(40 / 100)
        assert result.test == pytest.approx(20 / 100)

    def test_single_unassigned_item(self):
        """Test with just one unassigned item"""
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 7,
                DatasetItemSubset.VALIDATION: 2,
                DatasetItemSubset.TESTING: 0,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        result = distribution.compute_adjusted_ratios(target_ratios, unassigned_count=1)

        # Total will be 10, target: 7 train, 2 val, 1 test
        # Needed: 0 train, 0 val, 1 test
        # Single item should go to test
        assert result.train == pytest.approx(0.0)
        assert result.val == pytest.approx(0.0)
        assert result.test == pytest.approx(1.0)

    def test_unassigned_item_negative_value(self):
        """Test that negative unassigned_count raises ValueError"""
        distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 70,
                DatasetItemSubset.VALIDATION: 20,
                DatasetItemSubset.TESTING: 10,
            }
        )
        target_ratios = SplitRatios(train=0.7, val=0.2, test=0.1)

        with pytest.raises(ValueError, match=re.escape("Parameter 'unassigned_count' cannot be negative (got -10)")):
            distribution.compute_adjusted_ratios(target_ratios, unassigned_count=-10)
