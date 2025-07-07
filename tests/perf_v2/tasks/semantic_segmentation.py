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

from otx.types.task import OTXTaskType

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
        name="tiny_human_railway_animal",
        path=Path("semantic_seg/tiny_human_railway_animal_6_6_6"),
        group="tiny",
    ),
    DatasetInfo(
        name="tiny_cell_labels",
        path=Path("semantic_seg/tiny_cell_labels_6_6_6"),
        group="tiny",
    ),
    DatasetInfo(
        name="small_satellite_buildings",
        path=Path("semantic_seg/small_satellite_buildings_20_8_12"),
        group="small",
    ),
    DatasetInfo(
        name="small_aerial",
        path=Path("semantic_seg/small_aerial_50_20_30"),
        group="small",
    ),
    DatasetInfo(
        name="medium_kitti",
        path=Path("semantic_seg/medium_kitti_150_50_50"),
        group="medium",
    ),
    DatasetInfo(
        name="medium_voc_otx_cut",
        path=Path("semantic_seg/medium_voc_otx_cut_662_300_300"),
        group="medium",
    ),
    DatasetInfo(
        name="large_idd20k",
        path=Path("semantic_seg/large_idd20k_lite_1122_204_281"),
        group="large",
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="training:val/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/Dice", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
]
