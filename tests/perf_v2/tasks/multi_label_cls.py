# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX multi-label classification performance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.MULTI_LABEL_CLS


BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:val/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="train:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]

MODEL_TEST_CASES = [
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="deit_tiny", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="multilabel_tiny_bccd",
        path=Path("multilabel_classification/mlabel_tiny_bccd_24_6_100"),
        group="tiny",
    ),
    DatasetInfo(
        name="multilabel_small_coco",
        path=Path("multilabel_classification/mlabel_small_coco_80_20_100"),
        group="small",
    ),
    DatasetInfo(
        name="multilabel_medium_edsavehicle",
        path=Path("multilabel_classification/mlabel_medium_edsavehicle_600_150_200"),
        group="medium",
    ),
    DatasetInfo(
        name="multilabel_large_aid",
        path=Path("multilabel_classification/mlabel_large_aid_1000_300_300"),
        group="large",
    ),
]
