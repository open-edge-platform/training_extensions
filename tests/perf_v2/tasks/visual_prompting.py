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


# TODO (someone): Compare with DETECTION CRITERIA and fill in the missing values
BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="val/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="test/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="export/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/dice", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
]
