# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX anomaly performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.ANOMALY

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="padim", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="stfpm", category="accuracy"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="mvtec_wood_medium",
        path=Path("anomaly/mvtec/wood_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_hazelnut_large",
        path=Path("anomaly/mvtec/hazelnut_large"),
        group="large",
        extra_overrides={},
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="train/gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="test/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="export/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/image_F1Score", summary="max", compare=">", margin=0.1),
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
