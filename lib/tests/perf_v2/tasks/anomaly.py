# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX anomaly performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.ANOMALY

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="padim", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="uflow", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="stfpm", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="hazelnut_toy_tiny",
        path=Path("anomaly/hazelnut_tiny"),
        group="tiny",
    ),
    DatasetInfo(
        name="hazelnut_toy_small",
        path=Path("anomaly/hazelnut_small"),
        group="small",
    ),
    DatasetInfo(
        name="mvtec_bottle_medium",
        path=Path("anomaly/mvtec/mvtec_bottle_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_cable_medium",
        path=Path("anomaly/mvtec/mvtec_cable_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_capsule_medium",
        path=Path("anomaly/mvtec/mvtec_capsule_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_carpet_medium",
        path=Path("anomaly/mvtec/mvtec_carpet_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_grid_medium",
        path=Path("anomaly/mvtec/mvtec_grid_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_hazelnut_medium",
        path=Path("anomaly/mvtec/mvtec_hazelnut_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_leather_medium",
        path=Path("anomaly/mvtec/mvtec_leather_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_metal_nut_medium",
        path=Path("anomaly/mvtec/mvtec_metal_nut_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_pill_medium",
        path=Path("anomaly/mvtec/mvtec_pill_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_screw_medium",
        path=Path("anomaly/mvtec/mvtec_screw_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_tile_medium",
        path=Path("anomaly/mvtec/mvtec_tile_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_toothbrush_medium",
        path=Path("anomaly/mvtec/mvtec_toothbrush_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_transistor_medium",
        path=Path("anomaly/mvtec/mvtec_transistor_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_wood_medium",
        path=Path("anomaly/mvtec/mvtec_wood_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="mvtec_zipper_medium",
        path=Path("anomaly/mvtec/mvtec_zipper_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="visa_candle_large",
        path=Path("anomaly/visa/visa_candle_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_capsules_medium",
        path=Path("anomaly/visa/visa_capsules_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="visa_cashew_medium",
        path=Path("anomaly/visa/visa_cashew_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="visa_chewinggum_medium",
        path=Path("anomaly/visa/visa_chewinggum_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="visa_fryum_medium",
        path=Path("anomaly/visa/visa_fryum_medium"),
        group="medium",
    ),
    DatasetInfo(
        name="visa_macaroni1_large",
        path=Path("anomaly/visa/visa_macaroni1_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_macaroni2_large",
        path=Path("anomaly/visa/visa_macaroni2_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_pcb1_large",
        path=Path("anomaly/visa/visa_pcb1_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_pcb2_large",
        path=Path("anomaly/visa/visa_pcb2_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_pcb3_large",
        path=Path("anomaly/visa/visa_pcb3_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_pcb4_large",
        path=Path("anomaly/visa/visa_pcb4_large"),
        group="large",
    ),
    DatasetInfo(
        name="visa_pipe_fryum_medium",
        path=Path("anomaly/visa/visa_pipe_fryum_medium"),
        group="medium",
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
