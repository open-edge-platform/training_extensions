# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX performance benchmark tests."""

from otx.types.task import OTXTaskType

from .tasks import (
    anomaly,
    detection,
    h_label_cls,
    instance_segmentation,
    keypoint_detection,
    multi_class_cls,
    multi_label_cls,
    semantic_segmentation,
)

CRITERIA_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.BENCHMARK_CRITERIA,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.BENCHMARK_CRITERIA,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.BENCHMARK_CRITERIA,
    OTXTaskType.ANOMALY: anomaly.BENCHMARK_CRITERIA,
    OTXTaskType.MULTI_CLASS_CLS: multi_class_cls.BENCHMARK_CRITERIA,
    OTXTaskType.MULTI_LABEL_CLS: multi_label_cls.BENCHMARK_CRITERIA,
    OTXTaskType.H_LABEL_CLS: h_label_cls.BENCHMARK_CRITERIA,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.BENCHMARK_CRITERIA,
}

MODEL_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.MODEL_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.MODEL_TEST_CASES,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.MODEL_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.MODEL_TEST_CASES,
    OTXTaskType.MULTI_CLASS_CLS: multi_class_cls.MODEL_TEST_CASES,
    OTXTaskType.MULTI_LABEL_CLS: multi_label_cls.MODEL_TEST_CASES,
    OTXTaskType.H_LABEL_CLS: h_label_cls.MODEL_TEST_CASES,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.MODEL_TEST_CASES,
}

DATASET_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.DATASET_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.DATASET_TEST_CASES,
    OTXTaskType.SEMANTIC_SEGMENTATION: semantic_segmentation.DATASET_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.DATASET_TEST_CASES,
    OTXTaskType.MULTI_CLASS_CLS: multi_class_cls.DATASET_TEST_CASES,
    OTXTaskType.MULTI_LABEL_CLS: multi_label_cls.DATASET_TEST_CASES,
    OTXTaskType.H_LABEL_CLS: h_label_cls.DATASET_TEST_CASES,
    OTXTaskType.KEYPOINT_DETECTION: keypoint_detection.DATASET_TEST_CASES,
}


TASK_METRIC_MAP = {
    OTXTaskType.ANOMALY: "image_F1Score",  # perf v2 uses single anomaly task
    OTXTaskType.MULTI_CLASS_CLS: "accuracy",
    OTXTaskType.MULTI_LABEL_CLS: "accuracy",
    OTXTaskType.H_LABEL_CLS: "accuracy",
    OTXTaskType.DETECTION: "f1-score",
    OTXTaskType.INSTANCE_SEGMENTATION: "f1-score",
    OTXTaskType.SEMANTIC_SEGMENTATION: "Dice",
    OTXTaskType.KEYPOINT_DETECTION: "PCK",
}


METADATA_ENTRIES = [
    "date",
    "task",
    "model",
    "data_group",
    "data",
    "otx_version",
    "otx_ref",
    "test_branch",
    "test_commit",
    "cpu_info",
    "accelerator_info",
    "user_name",
    "machine_name",
]
