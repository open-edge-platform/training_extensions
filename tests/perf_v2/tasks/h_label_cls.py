# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX hierarchical classification performance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.H_LABEL_CLS

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
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="deit_tiny", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="hlabel_tiny_playingcards",
        path=Path("hlabel_classification/hlabel_tiny_playingcards-2L-6N_36_20_100"),
        group="tiny",
    ),
    DatasetInfo(
        name="hlabel_small_cub",
        path=Path("hlabel_classification/hlabel_small_cub-3L-6N_72_24_100"),
        group="small",
    ),
    DatasetInfo(
        name="hlabel_medium_stanfordcars",
        path=Path("hlabel_classification/hlabel_medium_stanfordcars-26N-3L_350_50_200"),
        group="medium",
    ),
    DatasetInfo(
        name="hlabel_large_plantdiseases",
        path=Path("hlabel_classification/hlabel_large_plantdiseases-32N-5L_1000_300_300"),
        group="large",
    ),
]
