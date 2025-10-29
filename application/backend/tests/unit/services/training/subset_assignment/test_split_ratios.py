# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.models import DatasetItemSubset
from app.services.training.subset_assignment import SplitRatios


class TestSplitRatios:
    """Test cases for SplitRatios value object"""

    def test_valid_split_ratios(self):
        """Test creation of valid split ratios"""
        split = SplitRatios(train=0.7, val=0.2, test=0.1)
        assert split.train == 0.7
        assert split.val == 0.2
        assert split.test == 0.1

    @pytest.mark.parametrize(
        "train,val,test",
        [
            (0.5, 0.2, 0.2),  # 0.9
            (0.6, 0.3, 0.2),  # 1.1
        ],
    )
    def test_split_ratios_sum_validation(self, train, val, test):
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            SplitRatios(train=train, val=val, test=test)

    def test_split_ratios_boundary_low(self):
        """Test that split ratios at 0.99 boundary are accepted"""
        split = SplitRatios(train=0.59, val=0.2, test=0.2)
        assert split.train + split.val + split.test == 0.99

    def test_split_ratios_boundary_high(self):
        """Test that split ratios at 1.01 boundary are accepted"""
        split = SplitRatios(train=0.61, val=0.2, test=0.2)
        assert split.train + split.val + split.test == 1.01

    @pytest.mark.parametrize(
        "target,train,val,test",
        [
            (100, 70, 20, 10),  # basic
            (10, 7, 2, 1),  # small numbers
            (50, 35, 10, 5),  # half size
            (97, 67, 19, 11),  # rounding case
            (1, 0, 0, 1),  # single item
        ],
    )
    def test_to_fold_sizes(self, target, train, val, test):
        """Test conversion of split ratios to absolute fold sizes"""
        split = SplitRatios(train=0.7, val=0.2, test=0.1)
        fold_sizes = split.to_fold_sizes(target)

        assert fold_sizes[DatasetItemSubset.TRAINING] == train
        assert fold_sizes[DatasetItemSubset.VALIDATION] == val
        assert fold_sizes[DatasetItemSubset.TESTING] == test
        assert sum(fold_sizes.values()) == target

    def test_to_list(self):
        """Test conversion of split ratios to list"""
        split = SplitRatios(train=0.7, val=0.2, test=0.1)
        ratios_list = split.to_list()

        assert ratios_list == [0.7, 0.2, 0.1]
        assert len(ratios_list) == 3

    def test_all_train(self):
        """Test split with all items in training"""
        split = SplitRatios(train=1.0, val=0.0, test=0.0)
        fold_sizes = split.to_fold_sizes(100)

        assert fold_sizes[DatasetItemSubset.TRAINING] == 100
        assert fold_sizes[DatasetItemSubset.VALIDATION] == 0
        assert fold_sizes[DatasetItemSubset.TESTING] == 0

    def test_equal_split(self):
        """Test equal split across all subsets"""
        split = SplitRatios(train=0.33, val=0.33, test=0.34)
        fold_sizes = split.to_fold_sizes(300)

        assert fold_sizes[DatasetItemSubset.TRAINING] == 99
        assert fold_sizes[DatasetItemSubset.VALIDATION] == 99
        # Test gets the remainder to ensure total is preserved
        assert fold_sizes[DatasetItemSubset.TESTING] == 102
        assert sum(fold_sizes.values()) == 300
