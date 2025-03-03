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
