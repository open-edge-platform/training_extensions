# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for IoU computation and linear assignment."""

import numpy as np
import pytest

from getitrack.matching.iou import (
    fuse_score,
    iou_matrix,
    linear_assignment,
)


class TestIoUMatrix:
    def test_half_overlap_value(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 0, 15, 10]], dtype=np.float32)
        # Intersection 50, union 150, IoU = 1/3.
        assert iou_matrix(a, b)[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_empty_input_returns_empty_matrix(self):
        empty = np.empty((0, 4), dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert iou_matrix(empty, boxes).shape == (0, 1)
        assert iou_matrix(boxes, empty).shape == (1, 0)

    def test_bad_shape_raises(self):
        bad = np.zeros((3, 5), dtype=np.float32)
        good = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="boxes_a must have shape"):
            iou_matrix(bad, good)

    def test_nan_input_raises(self):
        nan_box = np.array([[0.0, 0.0, np.nan, 10.0]], dtype=np.float32)
        good = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="boxes_a contains non-finite"):
            iou_matrix(nan_box, good)
        with pytest.raises(ValueError, match="boxes_b contains non-finite"):
            iou_matrix(good, nan_box)


class TestFuseScore:
    def test_high_score_lowers_cost(self):
        cost = np.array([[0.5, 0.5]], dtype=np.float32)
        scores = np.array([1.0, 0.1], dtype=np.float32)
        fused = fuse_score(cost, scores)
        assert fused[0, 0] < fused[0, 1]


class TestLinearAssignment:
    def test_empty_cost_matrix(self):
        cost = np.empty((0, 3), dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.shape == (0, 2)
        assert ua.tolist() == []
        assert ub.tolist() == [0, 1, 2]

    def test_over_threshold_pairs_excluded_from_solution(self):
        # Unconstrained Hungarian would pick (0->1, 1->0); gating the over-threshold
        # (0,1) pair first yields the valid optimum (0->0), leaving row 1 unmatched.
        cost = np.array([[0.1, 0.51], [0.5, 100.0]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.tolist() == [[0, 0]]
        assert ua.tolist() == [1]
        assert ub.tolist() == [1]
