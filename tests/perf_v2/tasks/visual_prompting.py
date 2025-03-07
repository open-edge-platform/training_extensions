# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX visual prompting perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.VISUAL_PROMPTING

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="sam_tiny_vit", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="sam_vit_b", category="accuracy"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name=f"wgisd_small_{idx}",
        path=Path("visual_prompting/wgisd_small") / f"{idx}",
        group="small",
        extra_overrides={},
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="coco_car_person_medium",
        path=Path("visual_prompting/coco_car_person_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="vitens_coliform",
        path=Path("visual_prompting/Vitens-Coliform-coco"),
        group="large",
        extra_overrides={},
    ),
]


# TODO (someone): align with detection task (adding gpu_mem, latency, optimize/e2e, etc)
BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
