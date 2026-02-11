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
        "target,train,val,test,ratios",
        [
            (100, 70, 20, 10, (0.7, 0.2, 0.1)),  # basic
            (97, 68, 19, 10, (0.7, 0.2, 0.1)),  # rounding case
            (300, 99, 99, 102, (0.33, 0.33, 0.34)),  # equal ratios with rounding
            (10, 4, 3, 3, (0.34, 0.33, 0.33)),  # equal ratios with rounding
            (4, 2, 1, 1, (0.5, 0.5, 0)),  # reduction priority when val and train are equal
            (11, 7, 3, 1, (0.6, 0.3, 0.1)),  # reduction from the largest
            (3, 1, 1, 1, (0.8, 0.15, 0.05)),  # each fold gets at least 1 item, even if ratios are skewed
            (3, 1, 1, 1, (1.0, 0.0, 0.0)),  # all train, but still need to assign 1 to val and test
        ],
    )
    def test_to_fold_sizes(self, target: int, train: int, val: int, test: int, ratios: tuple[float, ...]):
        """Test conversion of split ratios to absolute fold sizes"""
        split = SplitRatios(*ratios)
        fold_sizes = split.to_fold_sizes(target)

        assert fold_sizes[DatasetItemSubset.TRAINING] == train
        assert fold_sizes[DatasetItemSubset.VALIDATION] == val
        assert fold_sizes[DatasetItemSubset.TESTING] == test
        assert sum(fold_sizes.values()) == target

    def test_to_fold_sizes_boundary_low(self):
        """Test that to_fold_sizes raises error for less than 3 items"""
        split = SplitRatios(train=0.7, val=0.2, test=0.1)
        with pytest.raises(ValueError, match="Need at least 3 items to create all subsets"):
            split.to_fold_sizes(2)

    def test_to_list(self):
        """Test conversion of split ratios to list"""
        split = SplitRatios(train=0.7, val=0.2, test=0.1)
        ratios_list = split.to_list()

        assert ratios_list == [0.7, 0.2, 0.1]
        assert len(ratios_list) == 3
