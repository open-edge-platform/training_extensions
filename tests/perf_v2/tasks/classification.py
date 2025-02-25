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

from otx.core.types.task import OTXTaskType

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
        name=f"multiclass_CUB_small_{idx}",
        path=Path("multiclass_classification/multiclass_CUB_small") / f"{idx}",
        group="small",
        extra_overrides={},
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="multiclass_CUB_medium",
        path=Path("multiclass_classification/multiclass_CUB_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="multiclass_food20_large",
        path=Path("multiclass_classification/multiclass_food20_large"),
        group="large",
        extra_overrides={},
    ),
]

# TODO (someone): Compare with DETECTION CRITERIA and fill in the missing values
MULTI_CLASS_BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
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
        name=f"multilabel_CUB_small_{idx}",
        path=Path("multilabel_classification/multilabel_CUB_small") / f"{idx}",
        group="small",
        extra_overrides={},
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="multilabel_CUB_medium",
        path=Path("multilabel_classification/multilabel_CUB_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="multilabel_food20_large",
        path=Path("multilabel_classification/multilabel_food20_large"),
        group="large",
        extra_overrides={},
    ),
]

# TODO (someone): Compare with DETECTION CRITERIA and fill in the missing values
MULTI_LABEL_BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
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
        name=f"hlabel_CUB_small_{idx}",
        path=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
        group="small",
        extra_overrides={},
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="hlabel_CUB_medium",
        path=Path("hlabel_classification/hlabel_CUB_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="cifar100_label_group_datum_format_large",
        path=Path("hlabel_classification/cifar100_label_group_datum_format_large"),
        group="large",
        extra_overrides={},
    ),
]

# TODO (someone): Compare with DETECTION CRITERIA and fill in the missing values
H_LABEL_CLS_BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
]
