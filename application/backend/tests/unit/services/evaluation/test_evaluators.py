# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest

from app.services.evaluation import (
    AveragingMethod,
    DetectionEvaluator,
    InstanceSegmentationEvaluator,
    MultiClassClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)

logger = logging.getLogger(__name__)


class TestMultiClassClassificationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_multiclass_classification_dataset_gt, fxt_multiclass_classification_dataset_pred
    ) -> None:
        evaluator = MultiClassClassificationEvaluator(
            predictions_dataset=fxt_multiclass_classification_dataset_pred,
            ground_truth_dataset=fxt_multiclass_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()
        confusion_matrix = evaluator.confusion_matrix()

        assert accuracy == pytest.approx(3 / 5)  # 3 correct out of 5
        # In multiclass classification, micro precision, recall, micro F1 and accuracy are equal
        assert precision_micro == recall_micro == f1_score_micro == accuracy
        assert precision_macro == pytest.approx(2 / 3)  # (1 + 0.5 + 0.5) / 3
        assert recall_macro == pytest.approx(2 / 3)
        assert f1_score_macro == pytest.approx(2 / 3)
        assert precision_weighted == pytest.approx(3 / 5)  # (1*1 + 0.5*2 + 0.5*2) / 5
        assert recall_weighted == pytest.approx(3 / 5)
        assert f1_score_weighted == pytest.approx(3 / 5)
        assert (confusion_matrix == np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])).all()

    def test_evaluate_with_perfect_predictions(self, fxt_multiclass_classification_dataset_pred) -> None:
        # Using the same dataset for GT and predictions to simulate perfect predictions
        evaluator = MultiClassClassificationEvaluator(
            predictions_dataset=fxt_multiclass_classification_dataset_pred,
            ground_truth_dataset=fxt_multiclass_classification_dataset_pred,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()
        confusion_matrix = evaluator.confusion_matrix()

        assert accuracy == 1.0
        assert precision_micro == precision_macro == precision_weighted == 1.0
        assert recall_micro == recall_macro == recall_weighted == 1.0
        assert f1_score_micro == f1_score_macro == f1_score_weighted == 1.0
        assert np.trace(confusion_matrix) == confusion_matrix.sum()  # all non-zero values are on the diagonal


class TestMultiLabelClassificationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_multilabel_classification_dataset_gt, fxt_multilabel_classification_dataset_pred
    ) -> None:
        evaluator = MultiLabelClassificationEvaluator(
            predictions_dataset=fxt_multilabel_classification_dataset_pred,
            ground_truth_dataset=fxt_multilabel_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()

        assert accuracy == pytest.approx(1 / 3)  # only one sample is completely correct (subset accuracy)
        assert precision_micro == pytest.approx(4 / 5)  # 4 correct labels out of 5 predicted
        assert precision_macro == pytest.approx(5 / 6)  # (2/2 + 1/1 + 1/2) / 3
        assert precision_weighted == pytest.approx(9 / 10)  # (1*2 + 1*2 + 0.5*0.5) / 5
        assert recall_micro == pytest.approx(4 / 5)  # 4 ground truth labels out of 5 found
        assert recall_macro == pytest.approx(5 / 6)  # (2/2 + 1/2 + 1/1) / 3
        assert recall_weighted == recall_micro  # always identical to micro recall
        assert f1_score_micro == pytest.approx(4 / 5)  # harmonic mean of micro precision and recall
        assert f1_score_macro == pytest.approx(7 / 9)  # harmonic mean of macro precision and recall
        assert f1_score_weighted == pytest.approx(4 / 5)  # harmonic mean of weighted precision and recall

    def test_evaluate_with_perfect_predictions(self, fxt_multilabel_classification_dataset_pred) -> None:
        # Using the same dataset for GT and predictions to simulate perfect predictions
        evaluator = MultiLabelClassificationEvaluator(
            predictions_dataset=fxt_multilabel_classification_dataset_pred,
            ground_truth_dataset=fxt_multilabel_classification_dataset_pred,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()

        assert accuracy == 1.0
        assert precision_micro == precision_macro == precision_weighted == 1.0
        assert recall_micro == recall_macro == recall_weighted == 1.0
        assert f1_score_micro == f1_score_macro == f1_score_weighted == 1.0


class TestDetectionEvaluator:
    def test_evaluate_with_imperfect_predictions(self, fxt_detection_dataset_gt, fxt_detection_dataset_pred) -> None:
        """Evaluate on a scenario where some predictions are correct and some are incorrect."""
        evaluator = DetectionEvaluator(
            predictions_dataset=fxt_detection_dataset_pred,
            ground_truth_dataset=fxt_detection_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        # Example is constructed with a bbox with 60% IoU, that's between the 50% and 75% thresholds
        assert 0 < map_dict["AP_75"] < map_dict["AP_50"] < 1
        assert 0 < map_dict["AP_all"] < 1

    def test_evaluate_with_perfect_predictions(self, fxt_detection_dataset_pred) -> None:
        """Evaluate on a scenario where all predictions are correct."""
        # Using the same dataset for GT and predictions to simulate perfect predictions
        evaluator = DetectionEvaluator(
            predictions_dataset=fxt_detection_dataset_pred,
            ground_truth_dataset=fxt_detection_dataset_pred,
        )

        map_dict = evaluator.mean_average_precision()

        assert map_dict["AP_50"] == map_dict["AP_75"] == map_dict["AP_all"] == 1.0


class TestInstanceSegmentationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_instance_segmentation_dataset_gt, fxt_instance_segmentation_dataset_pred
    ) -> None:
        """Evaluate on a scenario where some predictions are correct and some are incorrect."""
        evaluator = InstanceSegmentationEvaluator(
            predictions_dataset=fxt_instance_segmentation_dataset_pred,
            ground_truth_dataset=fxt_instance_segmentation_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        # Example is constructed with a polygon with 64% IoU, that's between the 50% and 75% thresholds
        assert 0 < map_dict["AP_75"] < map_dict["AP_50"] < 1
        assert 0 < map_dict["AP_all"] < 1

    def test_evaluate_with_perfect_predictions(self, fxt_instance_segmentation_dataset_pred) -> None:
        """Evaluate on a scenario where all predictions are correct."""
        # Using the same dataset for GT and predictions to simulate perfect predictions
        evaluator = InstanceSegmentationEvaluator(
            predictions_dataset=fxt_instance_segmentation_dataset_pred,
            ground_truth_dataset=fxt_instance_segmentation_dataset_pred,
        )

        map_dict = evaluator.mean_average_precision()

        assert map_dict["AP_50"] == map_dict["AP_75"] == map_dict["AP_all"] == 1.0
