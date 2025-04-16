# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX classification perfomance benchmark tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> bool:
    """Override the deterministic setting for classification tasks."""
    deterministic = request.config.getoption("--deterministic")
    deterministic = True if deterministic is None else deterministic == "true"
    log.info(f"{deterministic=}")
    return deterministic


class TestPerfSingleLabelClassification(PerfTestBase):
    """Benchmark single-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="deit_tiny", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="dino_v2", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_b3", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_v2_l", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_mobilenet_v3_small", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name="multiclass_tiny_pneumonia",
            path=Path("multiclass_classification/mcls_tiny_pneumonia_12_6_200"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_tiny_cub_woodpecker",
            path=Path("multiclass_classification/mcls_tiny_cub_woodpecker_24_12_200"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_small_flowers",
            path=Path("multiclass_classification/mcls_small_flowers_60_12_200"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_small_eurosat",
            path=Path("multiclass_classification/mcls_small_eurosat_80_40_200"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_medium_resisc",
            path=Path("multiclass_classification/mcls_medium_resisc_500_100_400"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_large_cub100",
            path=Path("multiclass_classification/mcls_large_cub100_3764_900_1200"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),

    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
    ]

    @pytest.mark.parametrize(
        "fxt_model",
        MODEL_TEST_CASES,
        ids=lambda model: model.name,
        indirect=True,
    )
    @pytest.mark.parametrize(
        "fxt_dataset",
        DATASET_TEST_CASES,
        ids=lambda dataset: dataset.name,
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
        fxt_accelerator: str,
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfMultiLabelClassification(PerfTestBase):
    """Benchmark multi-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_label_cls", name="deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name="multilabel_tiny_bccd",
            path=Path("multilabel_classification/mlabel_tiny_bccd_24_6_100"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_small_coco",
            path=Path("multilabel_classification/mlabel_small_coco_80_20_100"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_medium_edsavehicle",
            path=Path("multilabel_classification/mlabel_medium_edsavehicle_600_150_200"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_large_aid",
            path=Path("multilabel_classification/mlabel_large_aid_1000_300_300"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
    ]

    @pytest.mark.parametrize(
        "fxt_model",
        MODEL_TEST_CASES,
        ids=lambda model: model.name,
        indirect=True,
    )
    @pytest.mark.parametrize(
        "fxt_dataset",
        DATASET_TEST_CASES,
        ids=lambda dataset: dataset.name,
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
        fxt_accelerator: str,
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfHierarchicalLabelClassification(PerfTestBase):
    """Benchmark hierarchical-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/h_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/h_label_cls", name="deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name="hlabel_tiny_playingcards",
            path=Path("hlabel_classification/hlabel_tiny_playingcards-2L-6N_36_20_100"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="hlabel_small_cub",
            path=Path("hlabel_classification/hlabel_small_cub-3L-6N_72_24_100"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="hlabel_medium_stanfordcars",
            path=Path("hlabel_classification/hlabel_medium_stanfordcars-26N-3L_350_50_200"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="hlabel_large_plantdiseases",
            path=Path("hlabel_classification/hlabel_large_plantdiseases-32N-5L_1000_300_300"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
    ]

    @pytest.mark.parametrize(
        "fxt_model",
        MODEL_TEST_CASES,
        ids=lambda model: model.name,
        indirect=True,
    )
    @pytest.mark.parametrize(
        "fxt_dataset",
        DATASET_TEST_CASES,
        ids=lambda dataset: dataset.name,
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
        fxt_accelerator: str,
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )
