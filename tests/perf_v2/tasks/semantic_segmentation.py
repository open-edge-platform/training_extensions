# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX semantic segmentation performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.SEMANTIC_SEGMENTATION

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="litehrnet_18", category="balance"),
    ModelInfo(task=TASK_TYPE.value, name="litehrnet_s", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="litehrnet_x", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="segnext_b", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="segnext_s", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="segnext_t", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="dino_v2", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name=f"kvasir_small_{idx}",
        path=Path("semantic_seg/kvasir_small") / f"{idx}",
        group="small",
        extra_overrides={},
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="cityscapes_185_70_medium",
        path=Path("semantic_seg/cityscapes_185_70_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="voc_2012_cut_large",
        path=Path("semantic_seg/voc_2012_cut_large"),
        group="large",
        extra_overrides={},
    ),
]

# TODO (someone): align with detection task (adding gpu_mem, latency, optimize/e2e, etc)
BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
