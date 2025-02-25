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
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="train/gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(torch)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(export)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(optimize)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(torch)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
]
