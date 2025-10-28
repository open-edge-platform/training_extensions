# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.experimental import Dataset

from app.services.evaluation import (
    DetectionEvaluator,
    EvaluatorFactory,
    InstanceSegmentationEvaluator,
    MultiClassClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)


class TestEvaluatorFactory:
    @pytest.mark.parametrize(
        "task_type,gt_dataset_fixture,pred_dataset_fixture",
        [
            (
                "multiclass_classification",
                "fxt_multiclass_classification_dataset_gt",
                "fxt_multiclass_classification_dataset_pred",
            ),
            (
                "multilabel_classification",
                "fxt_multilabel_classification_dataset_gt",
                "fxt_multilabel_classification_dataset_pred",
            ),
            ("detection", "fxt_detection_dataset_gt", "fxt_detection_dataset_pred"),
            ("instance_segmentation", "fxt_instance_segmentation_dataset_gt", "fxt_instance_segmentation_dataset_pred"),
        ],
        ids=["multiclass_classification", "multilabel_classification", "detection", "instance_segmentation"],
    )
    def test_get_evaluator(self, request, task_type, gt_dataset_fixture, pred_dataset_fixture) -> None:
        ground_truth_dataset: Dataset = request.getfixturevalue(gt_dataset_fixture)
        prediction_dataset: Dataset = request.getfixturevalue(pred_dataset_fixture)

        # Retrieve evaluator for classification task
        evaluator = EvaluatorFactory.get_evaluator(
            predictions_dataset=prediction_dataset, ground_truth_dataset=ground_truth_dataset
        )

        match task_type:
            case "multiclass_classification":
                assert isinstance(evaluator, MultiClassClassificationEvaluator)
            case "multilabel_classification":
                assert isinstance(evaluator, MultiLabelClassificationEvaluator)
            case "detection":
                assert isinstance(evaluator, DetectionEvaluator)
            case "instance_segmentation":
                assert isinstance(evaluator, InstanceSegmentationEvaluator)
            case _:
                raise AssertionError(f"Unrecognized task type {task_type}")
