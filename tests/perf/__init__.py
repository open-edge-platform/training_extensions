# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX perfomance benchmark tests."""


from otx.core.types.task import OTXTaskType

from .tasks import anomaly, detection, instance_segmentation

CRITERIA_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.BENCHMARK_CRITERIA,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.BENCHMARK_CRITERIA,
    OTXTaskType.ANOMALY: anomaly.BENCHMARK_CRITERIA,
}

MODEL_TEST_CASES_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.MODEL_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.MODEL_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.MODEL_TEST_CASES,
}

DATASET_TEST_CASES_COLLECTIONS = {
    OTXTaskType.DETECTION: detection.DATASET_TEST_CASES,
    OTXTaskType.INSTANCE_SEGMENTATION: instance_segmentation.DATASET_TEST_CASES,
    OTXTaskType.ANOMALY: anomaly.DATASET_TEST_CASES,
}
