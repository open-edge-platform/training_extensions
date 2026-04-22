# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune performance benchmark tests."""

from getitune.types.task import TaskType

from .tasks import (
    classification,
    detection,
    instance_segmentation,
    keypoint_detection,
    semantic_segmentation,
)

CRITERIA_COLLECTIONS = {
    TaskType.DETECTION: detection.BENCHMARK_CRITERIA,
    TaskType.INSTANCE_SEGMENTATION: instance_segmentation.BENCHMARK_CRITERIA,
    TaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.BENCHMARK_CRITERIA,
    TaskType.MULTI_CLASS_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    TaskType.MULTI_LABEL_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    TaskType.H_LABEL_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    TaskType.KEYPOINT_DETECTION: keypoint_detection.BENCHMARK_CRITERIA,
}

MODEL_COLLECTIONS = {
    TaskType.DETECTION: detection.MODEL_TEST_CASES,
    TaskType.INSTANCE_SEGMENTATION: instance_segmentation.MODEL_TEST_CASES,
    TaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.MODEL_TEST_CASES,
    TaskType.MULTI_CLASS_CLS: classification.MULTI_CLASS_MODEL_TEST_CASES,
    TaskType.MULTI_LABEL_CLS: classification.MULTI_LABEL_MODEL_TEST_CASES,
    TaskType.H_LABEL_CLS: classification.H_LABEL_CLS_MODEL_TEST_CASES,
    TaskType.KEYPOINT_DETECTION: keypoint_detection.MODEL_TEST_CASES,
}

DATASET_COLLECTIONS = {
    TaskType.DETECTION: detection.DATASET_TEST_CASES,
    TaskType.INSTANCE_SEGMENTATION: instance_segmentation.DATASET_TEST_CASES,
    TaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.DATASET_TEST_CASES,
    TaskType.MULTI_CLASS_CLS: classification.MULTI_CLASS_DATASET_TEST_CASES,
    TaskType.MULTI_LABEL_CLS: classification.MULTI_LABEL_DATASET_TEST_CASES,
    TaskType.H_LABEL_CLS: classification.H_LABEL_CLS_DATASET_TEST_CASES,
    TaskType.KEYPOINT_DETECTION: keypoint_detection.DATASET_TEST_CASES,
}
