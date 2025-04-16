# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX semantic segmentation perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfSemanticSegmentation(PerfTestBase):
    """Benchmark semantic segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_18", category="balance"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_s", category="speed"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_x", category="accuracy"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_b", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_s", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_t", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="dino_v2", category="other"),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name="tiny_human_railway_animal",
            path=Path("semantic_seg/tiny_human_railway_animal_6_6_6"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="tiny_cell_labels",
            path=Path("semantic_seg/tiny_cell_labels_6_6_6"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="small_satellite_buildings",
            path=Path("semantic_seg/small_satellite_buildings_20_8_12"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="small_aerial",
            path=Path("semantic_seg/small_aerial_50_20_30"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="medium_kitti",
            path=Path("semantic_seg/medium_kitti_150_50_50"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="medium_voc_otx_cut",
            path=Path("semantic_seg/medium_voc_otx_cut_662_300_300"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="large_idd20k",
            path=Path("semantic_seg/large_idd20k_lite_1122_204_281"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/Dice", summary="max", compare=">", margin=0.1),
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
