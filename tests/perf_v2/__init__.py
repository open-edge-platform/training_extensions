# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX performance benchmark tests."""


from otx.core.types.task import OTXTaskType

from .tasks import (
    anomaly,
    classification,
    detection,
    instance_segmentation,
    keypoint_detection,
    semantic_segmentation,
    visual_prompting,
)

CRITERIA_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.BENCHMARK_CRITERIA,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.BENCHMARK_CRITERIA,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.BENCHMARK_CRITERIA,
    OTXTaskType.ANOMALY: anomaly.BENCHMARK_CRITERIA,
    OTXTaskType.MULTI_CLASS_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    OTXTaskType.MULTI_LABEL_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    OTXTaskType.H_LABEL_CLS: classification.CLASSIFICATION_BENCHMARK_CRITERIA,
    OTXTaskType.VISUAL_PROMPTING: visual_prompting.BENCHMARK_CRITERIA,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.BENCHMARK_CRITERIA,
}

MODEL_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.MODEL_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.MODEL_TEST_CASES,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.MODEL_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.MODEL_TEST_CASES,
    OTXTaskType.MULTI_CLASS_CLS: classification.MULTI_CLASS_MODEL_TEST_CASES,
    OTXTaskType.MULTI_LABEL_CLS: classification.MULTI_LABEL_MODEL_TEST_CASES,
    OTXTaskType.H_LABEL_CLS: classification.H_LABEL_CLS_MODEL_TEST_CASES,
    OTXTaskType.VISUAL_PROMPTING: visual_prompting.MODEL_TEST_CASES,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.MODEL_TEST_CASES,
}

DATASET_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.DATASET_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.DATASET_TEST_CASES,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.DATASET_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.DATASET_TEST_CASES,
    OTXTaskType.MULTI_CLASS_CLS: classification.MULTI_CLASS_DATASET_TEST_CASES,
    OTXTaskType.MULTI_LABEL_CLS: classification.MULTI_LABEL_DATASET_TEST_CASES,
    OTXTaskType.H_LABEL_CLS: classification.H_LABEL_CLS_DATASET_TEST_CASES,
    OTXTaskType.VISUAL_PROMPTING: visual_prompting.DATASET_TEST_CASES,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.DATASET_TEST_CASES,
}
