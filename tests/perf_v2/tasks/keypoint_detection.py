# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OTX keypoint detection perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.core.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.KEYPOINT_DETECTION


MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="rtmpose_tiny_single_obj", category="speed"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="coco_person_keypoint_single_obj_small",
        path=Path("keypoint_detection/coco_keypoint_single_obj/small"),
        group="small",
        extra_overrides={},
    ),
    DatasetInfo(
        name="coco_person_keypoint_single_obj_medium",
        path=Path("keypoint_detection/coco_keypoint_single_obj/medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="coco_person_keypoint_single_obj_large",
        path=Path("keypoint_detection/coco_keypoint_single_obj/large"),
        group="large",
        extra_overrides={},
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="val/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="test/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="export/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
]
