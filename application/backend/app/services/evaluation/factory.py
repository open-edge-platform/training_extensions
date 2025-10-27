# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental import Dataset, Sample

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

    _registry: dict[type[Sample], type[Evaluator]] = {
        ClassificationSample: MultiClassClassificationEvaluator,
        MultilabelClassificationSample: MultiLabelClassificationEvaluator,
        DetectionSample: DetectionEvaluator,
        InstanceSegmentationSample: InstanceSegmentationEvaluator,
    }

    @classmethod
    def get_evaluator(cls, predictions_dataset: Dataset, ground_truth_dataset: Dataset) -> Evaluator:
        if predictions_dataset.dtype != ground_truth_dataset.dtype:
            raise ValueError("Predictions and ground truth datasets must have the same dtype")
        evaluator_cls = cls._registry.get(predictions_dataset.dtype)
        if evaluator_cls is None:
            raise NotImplementedError(f"Evaluator for dataset type {predictions_dataset.dtype} is not implemented")
        return evaluator_cls(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)
