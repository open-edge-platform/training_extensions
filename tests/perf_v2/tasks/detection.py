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
    # ModelInfo(task="detection", name="yolox_l", category="balance"),
    # ModelInfo(task="detection", name="dfine_x", category="accuracy"),
    # ModelInfo(task="detection", name="ssd_mobilenetv2", category="other"),
    # ModelInfo(task="detection", name="atss_resnext101", category="other"),
    # ModelInfo(task="detection", name="yolox_tiny", category="other"),
    # ModelInfo(task="detection", name="yolox_x", category="other"),
    # ModelInfo(task="detection", name="rtmdet_tiny", category="other"),
    # ModelInfo(task="detection", name="rtdetr_18", category="other"),
    # ModelInfo(task="detection", name="rtdetr_50", category="other"),
    # ModelInfo(task="detection", name="rtdetr_101", category="other"),
    # ModelInfo(task="detection", name="yolov9_s", category="other"),
    # ModelInfo(task="detection", name="yolov9_m", category="other"),
    # ModelInfo(task="detection", name="yolov9_c", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="wgisd_small_1",
        path=Path("detection/wgisd_small/1"),
        group="small",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
    DatasetInfo(
        name="wgisd_small_2",
        path=Path("detection/wgisd_small/2"),
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
