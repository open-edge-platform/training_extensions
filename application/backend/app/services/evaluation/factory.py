# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental import Dataset

from app.services.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)

from .evaluators import (
    DetectionEvaluator,
    Evaluator,
    InstanceSegmentationEvaluator,
    MultiClassClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)


class EvaluatorFactory:
    """Factory to get a suitable evaluator for a given set of ground truth and predictions datasets."""

    # TODO refactor with registry pattern

    @staticmethod
    def get_evaluator(predictions_dataset: Dataset, ground_truth_dataset: Dataset) -> Evaluator:
        if predictions_dataset.dtype != ground_truth_dataset.dtype:
            raise ValueError("Predictions and ground truth datasets must have the same dtype")

        if predictions_dataset.dtype is ClassificationSample:
            return MultiClassClassificationEvaluator(
                predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
            )
        if predictions_dataset.dtype is MultilabelClassificationSample:
            return MultiLabelClassificationEvaluator(
                predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
            )
        if predictions_dataset.dtype is DetectionSample:
            return DetectionEvaluator(
                predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
            )
        if predictions_dataset.dtype is InstanceSegmentationSample:
            return InstanceSegmentationEvaluator(
                predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
            )
        raise NotImplementedError(f"Evaluator for dataset type {predictions_dataset.dtype} is not implemented")
