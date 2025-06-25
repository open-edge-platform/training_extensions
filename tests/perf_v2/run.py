# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OTX Benchmark Entry Point."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from otx.types.task import OTXTaskType
from tests.perf_v2 import DATASET_COLLECTIONS, MODEL_COLLECTIONS
from tests.perf_v2.summary import load, summarize_task, task_high_level_summary
from tests.perf_v2.utils import (
    completeness_check,
    get_parser,
    setup_output_root,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
FAILED_JOBS_FILE = "failed_jobs.json"


def run_job(cmd: list[str], retries: int = MAX_RETRIES) -> dict | None:
    """Run a command and retry on failure.
    Args:
        cmd (list[str]): Command to run.
        retries (int): Number of retries on failure.
    Returns:
        dict | None: If the command fails, return a dictionary with error details.
                     If it succeeds, return None.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Running (attempt {attempt}): {' '.join(cmd)}")
            subprocess.run(cmd, check=True)  # noqa: S603
            return None  # Success  # noqa: TRY300
        except subprocess.CalledProcessError as e:  # noqa: PERF203
            stderr_output = ""
            try:
                subprocess.run(
                    cmd,  # noqa: S603
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e2:
                stderr_output = e2.stderr or ""
            logger.warning(f"Job failed on attempt {attempt}: {stderr_output.strip() or str(e)}")
            if attempt == retries:
                return {
                    "command": cmd,
                    "error": stderr_output.strip() or str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "retries": attempt,
                }
    return None


def load_failed_jobs(file_path: Path) -> list[list[str]]:
    """Load failed jobs from a JSON file."""
    if not file_path.exists():
        return []
    with file_path.open() as f:
        jobs = json.load(f)
        return [job["command"] for job in jobs]


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--resume-failed", action="store_true", help="Resume from failed_jobs.json")
    args = parser.parse_args()

    task_type = OTXTaskType[args.task]
    models = MODEL_COLLECTIONS[task_type]
    datasets = DATASET_COLLECTIONS[task_type]

    output_root = setup_output_root(args, task=task_type)

    failed_jobs = []

    if args.resume_failed:
        resume_jobs = load_failed_jobs(output_root / FAILED_JOBS_FILE)
        for cmd in resume_jobs:
            fail_result = run_job(cmd)
            if fail_result:
                failed_jobs.append(fail_result)
        # Clear failed_jobs.json if all resumed jobs succeeded
        if not failed_jobs:
            (output_root / FAILED_JOBS_FILE).write_text("[]")
    else:
        for model in models:
            for dataset in datasets:
                for seed in range(args.num_repeat):
                    if (output_root / model.name / dataset.name / str(seed)).exists():
                        logger.info(f"Skipping existing job for {model.name} on {dataset.name} with seed {seed}")
                        continue

                    cmd = [
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
                    ]
                    fail_result = run_job(cmd)
                    if fail_result:
                        failed_jobs.append(fail_result)

    if failed_jobs:
        with (output_root / FAILED_JOBS_FILE).open("w") as f:
            json.dump(failed_jobs, f, indent=2)
        logger.warning(f"{len(failed_jobs)} jobs failed. Details saved to {output_root / FAILED_JOBS_FILE}")

    raw_data = load(output_root)

    completeness_check(raw_data, models, datasets, num_repeat=args.num_repeat)

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
