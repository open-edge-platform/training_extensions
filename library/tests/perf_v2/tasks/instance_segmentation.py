# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX instance segmentation performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.INSTANCE_SEGMENTATION


MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rfdetr_seg_medium", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rfdetr_seg_large", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rfdetr_seg_small", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rfdetr_seg_xlarge", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_efficientnetb2b", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_swint", category="accuracy"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="wgisd_small",
        path=Path("detection/wgisd_merged_coco_small"),
        group="small",
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
