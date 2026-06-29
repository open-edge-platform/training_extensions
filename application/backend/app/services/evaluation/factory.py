# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental import Dataset

from .evaluators import (
    DetectionEvaluator,
    Evaluator,
    InstanceSegmentationEvaluator,
    MultiClassClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)


class EvaluatorFactory:
    """Factory to get a suitable evaluator for a given set of ground truth and predictions datasets."""

    @classmethod
    def get_evaluator(cls, predictions_dataset: Dataset, ground_truth_dataset: Dataset) -> Evaluator:
        if predictions_dataset.dtype != ground_truth_dataset.dtype:
            raise ValueError("Predictions and ground truth datasets must have the same dtype")

        from app.datumaro_converter import (
            DetectionTrainingSample,
            InstanceSegmentationTrainingSample,
            MulticlassClassificationTrainingSample,
            MultilabelClassificationTrainingSample,
        )

        match predictions_dataset.dtype:
            case t if t is MulticlassClassificationTrainingSample:
                evaluator_cls = MultiClassClassificationEvaluator
            case t if t is MultilabelClassificationTrainingSample:
                evaluator_cls = MultiLabelClassificationEvaluator
            case t if t is DetectionTrainingSample:
                evaluator_cls = DetectionEvaluator
            case t if t is InstanceSegmentationTrainingSample:
                evaluator_cls = InstanceSegmentationEvaluator
            case _:
                raise NotImplementedError(f"Evaluator for dataset type {predictions_dataset.dtype} is not implemented")

        return evaluator_cls(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)
