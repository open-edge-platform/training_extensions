# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX object detection performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.DETECTION

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="atss_mobilenetv2", category="default"),
    ModelInfo(task=TASK_TYPE.value, name="ssd_mobilenetv2", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="deim_dfine_x", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="deim_dfine_l", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="deim_dfine_m", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="deimv2_l", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="deimv2_m", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="deimv2_s", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtdetr_50", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rfdetr_large", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="rfdetr_medium", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="rfdetr_small", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_tiny", category="other"),
    # ModelInfo(task=TASK_TYPE.value, name="yolox_s", category="speed"),
    # ModelInfo(task=TASK_TYPE.value, name="yolox_l", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="yolox_x", category="other"),
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
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="training:val/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/map_50", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
