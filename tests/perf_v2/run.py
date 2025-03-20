# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OTX Benchmark Entry Point."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from otx.core.types.task import OTXTaskType
from tests.perf_v2 import DATASET_COLLECTIONS, MODEL_COLLECTIONS
from tests.perf_v2.summary import load, summarize_task, task_high_level_summary
from tests.perf_v2.utils import (
    completeness_check,
    current_date_str,
    get_parser,
    setup_output_root,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    task_type = OTXTaskType[args.task]
    models = MODEL_COLLECTIONS[task_type]
    datasets = DATASET_COLLECTIONS[task_type]

    current_date = current_date_str()
    output_root = setup_output_root(
        args,
        current_date,
        task=task_type,
    )

    for model in models:
        for dataset in datasets:
            for seed in range(args.num_repeat):
                subprocess.run(
                    [  # noqa: S603, S607
                        "python",
                        "-m",
                        "tests.perf_v2.benchmark",
                        "--task",
                        task_type.value,
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
                        "--user-name",
                        args.user_name,
                    ],
                    check=True,
                )

    raw_data = load(output_root)

    completeness_check(
        raw_data,
        models,
        datasets,
        num_repeat=args.num_repeat,
    )

    if len(raw_data):
        summary_file_root = Path(args.summary_file_root) if args.summary_file_root else output_root
        summary_file_root.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(summary_file_root / f"{task_type.value}-benchmark-raw-all.csv", index=False)
        logger.info(f"Saved merged raw data to {summary_file_root.resolve()!s}/{task_type.value}-benchmark-raw-all.csv")
        summarize_task(raw_data, task_type, summary_file_root)
        task_high_level_summary(raw_data, task_type, summary_file_root)
    else:
        msg = (
            f"{task_type.value} has no benchmark data loaded. "
            "Please check if the benchmark tests have been run successfully."
        )
        raise ValueError(msg)
