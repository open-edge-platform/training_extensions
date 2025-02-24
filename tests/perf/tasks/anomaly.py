# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX anomaly performance benchmark."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from otx.core.types.task import OTXTaskType

from tests.perf.utils import (
    get_parser,
    setup_output_root,
    current_date_str,
    DatasetInfo,
    ModelInfo,
    Criterion,
    completeness_check,
)


from tests.perf.summary import load, summarize_task
logger = logging.getLogger(__name__)


TASK_TYPE = OTXTaskType.ANOMALY

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="padim", category="speed"),
    ModelInfo(task=TASK_TYPE.value, name="stfpm", category="accuracy"),
]

DATASET_TEST_CASES = [
    DatasetInfo(
        name="mvtec_wood_medium",
        path=Path("anomaly/mvtec/wood_medium"),
        group="medium",
        extra_overrides={},
    ),
    DatasetInfo(
        name="mvtec_hazelnut_large",
        path=Path("anomaly/mvtec/hazelnut_large"),
        group="large",
        extra_overrides={},
    ),
]

BENCHMARK_CRITERIA = [ 
    Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="train/gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="test/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="export/image_F1Score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize/image_F1Score", summary="max", compare=">", margin=0.1),
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
        task=TASK_TYPE
    )

    for model in MODEL_TEST_CASES:
        for dataset in DATASET_TEST_CASES:
            for seed in range(args.num_repeat):
                subprocess.run(
                    [
                        "python",
                        "tests/perf/benchmark.py",
                        "--task", TASK_TYPE.value,
                        "--model", model.name,
                        "--dataset", dataset.name,
                        "--data-root", str(args.data_root),
                        "--output-root", str(output_root),
                        "--seed", str(seed),
                        "--num-epoch", str(args.num_epoch),
                        "--device", args.device,
                    ],
                    check=True,
                )

    raw_data = load(output_root)

    completeness_check(
        raw_data,
        MODEL_TEST_CASES,
        DATASET_TEST_CASES,
        num_repeat=args.num_repeat
    )

    if len(raw_data):
        output_root.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(output_root / f"{TASK_TYPE.value}-benchmark-raw-all.csv", index=False)
        logger.info("Saved merged raw data to", str(f"{TASK_TYPE.value}-benchmark-raw-all.csv"))
        summarize_task(raw_data, TASK_TYPE, output_root)
    else:
        logger.info("No data loaded. Please check if the benchmark tests have been run successfully.")
