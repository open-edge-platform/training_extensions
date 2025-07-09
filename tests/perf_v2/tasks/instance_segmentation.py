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
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_efficientnetb2b", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_swint", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtmdet_inst_tiny", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50_tv", category="other"),
]

DATASET_TEST_CASES = (
    [
        DatasetInfo(
            name=f"blueberry_tiny_{idx}",
            path=Path("instance_seg/blueberry_tiny_coco") / f"{idx}",
            group="tiny",
        )
        for idx in (1, 2, 3)
    ]
    + [
        DatasetInfo(
            name=f"wgisd_tiny_{idx}",
            path=Path("instance_seg/wgisd_merged_coco_tiny") / f"{idx}",
            group="tiny",
        )
        for idx in (1, 2, 3)
    ]
    + [
        DatasetInfo(
            name="skindetect",
            path=Path("instance_seg/skindetect-roboflow"),
            group="small",
        ),
        DatasetInfo(
            name="vitens_coliform",
            path=Path("instance_seg/Vitens-Coliform-coco"),
            group="small",
        ),
        DatasetInfo(
            name="Vitens-Aeromonas",
            path=Path("instance_seg/Vitens-Aeromonas-coco"),
            group="medium",
        ),
        DatasetInfo(
            name="Chicken",
            path=Path("instance_seg/Chicken-Real-Time-coco-roboflow"),
            group="medium",
        ),
        DatasetInfo(
            name="cityscapes",
            path=Path("instance_seg/cityscapes_coco_reduced"),
            group="large",
        ),
    ]
)

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/f1-score", summary="max", compare=">", margin=0.1),
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
