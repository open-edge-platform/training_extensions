# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX object detection performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.metrics.fmeasure import FMeasureCallable
from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.DETECTION

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="atss_mobilenetv2", category="default"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_s", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="dfine_x", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="atss_resnext101", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtdetr_101", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtdetr_18", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtdetr_50", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtmdet_tiny", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="ssd_mobilenetv2", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_tiny", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_l", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_x", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="pothole_small_1",
        path=Path("detection/pothole_small/1"),
        group="small",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
    DatasetInfo(
        name="pothole_small_2",
        path=Path("detection/pothole_small/2"),
        group="small",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="training:val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
