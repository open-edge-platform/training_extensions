# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX instance segmentation performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.metrics.fmeasure import FMeasureCallable
from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.INSTANCE_SEGMENTATION


MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_efficientnetb2b", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_swint", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtmdet_inst_tiny", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50_tv", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name=f"wgisd_small_{idx}",
        path=Path("instance_seg/wgisd_small") / f"{idx}",
        group="small",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="coco_car_person_medium",
        path=Path("instance_seg/coco_car_person_medium"),
        group="medium",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
    DatasetInfo(
        name="vitens_coliform",
        path=Path("instance_seg/Vitens-Coliform-coco"),
        group="large",
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
    Criterion(name="training:val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
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
