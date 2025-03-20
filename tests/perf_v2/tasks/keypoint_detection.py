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

# TODO (someone): align with detection task (adding gpu_mem, latency, optimize/e2e, etc)
BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/PCK", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
