# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX classification performance benchmark tests."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

CLASSIFICATION_BENCHMARK_CRITERIA = [
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


# ============= Multi-class classification =============

MULTI_CLASS_MODEL_TEST_CASES = [
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="deit_tiny", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="dino_v2", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_efficientnet_b3", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_efficientnet_v2_l", category="other"),
    ModelInfo(task=OTXTaskType.MULTI_CLASS_CLS.value, name="tv_mobilenet_v3_small", category="other"),
]

MULTI_CLASS_DATASET_TEST_CASES = [
    DatasetInfo(
        name="multiclass_tiny_pneumonia",
        path=Path("multiclass_classification/mcls_tiny_pneumonia_12_6_200"),
        group="tiny",
    ),
    DatasetInfo(
        name="multiclass_tiny_cub_woodpecker",
        path=Path("multiclass_classification/mcls_tiny_cub_woodpecker_24_12_200"),
        group="tiny",
    ),
    DatasetInfo(
        name="multiclass_small_flowers",
        path=Path("multiclass_classification/mcls_small_flowers_60_12_200"),
        group="small",
    ),
    DatasetInfo(
        name="multiclass_small_eurosat",
        path=Path("multiclass_classification/mcls_small_eurosat_80_40_200"),
        group="small",
    ),
    DatasetInfo(
        name="multiclass_medium_resisc",
        path=Path("multiclass_classification/mcls_medium_resisc_500_100_400"),
        group="medium",
    ),
    DatasetInfo(
        name="multiclass_large_cub100",
        path=Path("multiclass_classification/mcls_large_cub100_3764_900_1200"),
        group="large",
    ),
]

# ============= Multi-label classification =============
MULTI_LABEL_MODEL_TEST_CASES = [
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.MULTI_LABEL_CLS.value, name="deit_tiny", category="other"),
]

MULTI_LABEL_DATASET_TEST_CASES = [
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


# ============= Hierarchical-label classification =============


H_LABEL_CLS_MODEL_TEST_CASES = [
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="efficientnet_b0", category="speed"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="efficientnet_v2", category="balance"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="mobilenet_v3_large", category="accuracy"),
    ModelInfo(task=OTXTaskType.H_LABEL_CLS.value, name="deit_tiny", category="other"),
]

H_LABEL_CLS_DATASET_TEST_CASES = [
    DatasetInfo(
        name="hlabel_tiny_playingcards",
        path=Path("hlabel_classification/hlabel_tiny_playingcards-2L-6N_36_20_100"),
        group="tiny",
    ),
    DatasetInfo(
        name="hlabel_small_cub",
        path=Path("hlabel_classification/hlabel_small_cub-3L-6N_72_24_100"),
        group="small",
    ),
    DatasetInfo(
        name="hlabel_medium_stanfordcars",
        path=Path("hlabel_classification/hlabel_medium_stanfordcars-26N-3L_350_50_200"),
        group="medium",
    ),
    DatasetInfo(
        name="hlabel_large_plantdiseases",
        path=Path("hlabel_classification/hlabel_large_plantdiseases-32N-5L_1000_300_300"),
        group="large",
    ),
]
