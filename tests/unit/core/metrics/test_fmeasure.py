# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for OTX custom metrices."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from otx.core.metrics.fmeasure import FMeasure, get_n_false_negatives
from otx.core.types.label import LabelInfo


class TestFMeasure:
    @pytest.fixture()
    def fxt_preds(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.Tensor([[0.7, 0.6, 0.9, 0.6], [0.2, 0.5, 0.8, 0.6]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
            {
                "boxes": torch.Tensor([[0.3, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.5]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
        ]

    @pytest.fixture()
    def fxt_targets(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.Tensor([[0.8, 0.6, 0.9, 0.7], [0.3, 0.5, 0.8, 0.7]]),
                "labels": torch.IntTensor([0, 0]),
            },
            {
                "boxes": torch.Tensor([[0.4, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.4]]),
                "labels": torch.IntTensor([0, 0]),
            },
        ]

    def test_fmeasure(self, fxt_preds, fxt_targets) -> None:
        """Check whether f1 score is same with OTX1.x version."""
        metric = FMeasure(label_info=LabelInfo.from_num_classes(1))
        metric.update(fxt_preds, fxt_targets)
        result = metric.compute()
        assert result["f1-score"] == 0.5
        best_confidence_threshold = metric.best_confidence_threshold
        assert isinstance(best_confidence_threshold, float)

        metric.reset()
        assert metric.preds == []
        assert metric.targets == []

        # TODO(jaegukhyun): Add the following scenario
        # 1. Prepare preds and targets which can produce f1-score < 0.5
        # 2. Execute metric.compute()
        # 3. Assert best_confidence_threshold == metric.best_confidence_threshold

    def test_fmeasure_with_fixed_threshold(self, fxt_preds, fxt_targets) -> None:
        """Check fmeasure can compute f1 score given confidence threshold."""
        metric = FMeasure(label_info=LabelInfo.from_num_classes(1))

        metric.update(fxt_preds, fxt_targets)
        result = metric.compute(best_confidence_threshold=0.85)
        assert result["f1-score"] == 0.3333333432674408

    def test_get_fn(self):
        def _get_n_false_negatives_numpy(iou_matrix: np.ndarray, iou_threshold: float) -> int:
            n_false_negatives = 0
            for row in iou_matrix:
                if max(row) < iou_threshold:
                    n_false_negatives += 1
            for column in np.rot90(iou_matrix):
                indices = np.where(column > iou_threshold)
                n_false_negatives += max(len(indices[0]) - 1, 0)
            return n_false_negatives

        iou_matrix = torch.rand((10, 20))
        iou_threshold = np.random.rand()

        assert get_n_false_negatives(iou_matrix, iou_threshold) == _get_n_false_negatives_numpy(
            iou_matrix.numpy(),
            iou_threshold,
        )
