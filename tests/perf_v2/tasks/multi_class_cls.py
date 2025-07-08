# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX multi-class classification performance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.MULTI_CLASS_CLS

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="train:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]


MODEL_TEST_CASES = [
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="deit_tiny", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="dino_v2", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_efficientnet_b3", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_efficientnet_v2_l", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_mobilenet_v3_small", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="multiclass_tiny_pneumonia",
        path=Path("multiclass_classification/mcls_tiny_pneumonia_12_6_200"),
        group="tiny",
    ),
    DatasetInfo(
        name="multiclass_tiny_cub_woodpecker",
        path=Path("multiclass_classification/mcls_tiny_cub_woodpecker_24_12_200"),
        group="tiny",
    ),
    DatasetInfo(
        name="multiclass_small_flowers",
        path=Path("multiclass_classification/mcls_small_flowers_60_12_200"),
        group="small",
    ),
    DatasetInfo(
        name="multiclass_small_eurosat",
        path=Path("multiclass_classification/mcls_small_eurosat_80_40_200"),
        group="small",
    ),
    DatasetInfo(
        name="multiclass_medium_resisc",
        path=Path("multiclass_classification/mcls_medium_resisc_500_100_400"),
        group="medium",
    ),
    DatasetInfo(
        name="multiclass_large_cub100",
        path=Path("multiclass_classification/mcls_large_cub100_3764_900_1200"),
        group="large",
    ),
]
