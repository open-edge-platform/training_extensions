# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX instance segmentation performance benchmark."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from tests.perf.summary import load, summarize_task
from tests.perf.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
    completeness_check,
    current_date_str,
    get_parser,
    setup_output_root,
)

from otx.core.metrics.fmeasure import FMeasureCallable
from otx.core.types.task import OTXTaskType

logger = logging.getLogger(__name__)

TASK_TYPE = OTXTaskType.INSTANCE_SEGMENTATION


MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_efficientnetb2b", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_swint", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtmdet_inst_tiny", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="maskrcnn_r50_tv", category="other"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name=f"wgisd_small_{idx}",
        path=Path("instance_seg/wgisd_small") / f"{idx}",
        group="small",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    )
    for idx in (1, 2, 3)
] + [
    DatasetInfo(
        name="coco_car_person_medium",
        path=Path("instance_seg/coco_car_person_medium"),
        group="medium",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
    DatasetInfo(
        name="vitens_coliform",
        path=Path("instance_seg/Vitens-Coliform-coco"),
        group="large",
        extra_overrides={
            "test": {
                "metric": FMeasureCallable,
            },
        },
    ),
]

BENCHMARK_CRITERIA = [
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="train/gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize/e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(torch)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(export)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(optimize)/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="test(torch)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
]


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument(
        "--num-repeat",
        type=int,
        default=5,
        help="Number of repeated runs per model. Defaults to 5.",
    )

    args = parser.parse_args()
    current_date = current_date_str()
    output_root = setup_output_root(
        args,
        current_date,
        task=TASK_TYPE,
    )

    for model in MODEL_TEST_CASES:
        for dataset in DATASET_TEST_CASES:
            for seed in range(args.num_repeat):
                subprocess.run(
                    [
                        "python",
                        "tests/perf/benchmark.py",
                        "--task",
                        TASK_TYPE.value,
                        "--model",
                        model.name,
                        "--dataset",
                        dataset.name,
                        "--data-root",
                        str(args.data_root),
                        "--output-root",
                        str(output_root),
                        "--seed",
                        str(seed),
                        "--num-epoch",
                        str(args.num_epoch),
                        "--device",
                        args.device,
                    ],
                    check=True,
                )

    raw_data = load(output_root)

    completeness_check(
        raw_data,
        MODEL_TEST_CASES,
        DATASET_TEST_CASES,
        num_repeat=args.num_repeat,
    )

    if len(raw_data):
        output_root.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(output_root / f"{TASK_TYPE.value}-benchmark-raw-all.csv", index=False)
        logger.info(f"Saved merged raw data to {output_root!s}/{TASK_TYPE.value}-benchmark-raw-all.csv")
        summarize_task(raw_data, TASK_TYPE, output_root)
    else:
        logger.info("No data loaded. Please check if the benchmark tests have been run successfully.")
