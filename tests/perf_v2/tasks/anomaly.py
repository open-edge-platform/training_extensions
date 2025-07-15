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
        name="mvtec_cable_tiny",
        path=Path("anomaly/mvtec_cable_tiny"),
        group="tiny",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_capsule_tiny",
        path=Path("anomaly/mvtec_capsule_tiny"),
        group="tiny",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_hazelnut_small",
        path=Path("anomaly/mvtec_hazelnut_small"),
        group="small",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_screw_medium",
        path=Path("anomaly/mvtec_screw_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_transistor_small",
        path=Path("anomaly/mvtec_transistor_small"),
        group="small",
        extra_overrides={},
    ),
    DatasetInfo(
        name="visa_capsules_medium",
        path=Path("anomaly/visa_capsules_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="visa_pcb3_large",
        path=Path("anomaly/visa_pcb3_large"),
        group="large",
        extra_overrides={},
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
