#!/usr/bin/env python
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from cpuinfo import get_cpu_info
from jsonargparse import ArgumentParser, Namespace

import pandas as pd

from otx.core.types.task import OTXTaskType



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelInfo:
    """Benchmark model."""

    task: str
    name: str
    category: str

    def to_dict(self):
        return {"task": self.task, "name": self.name, "category": self.category}

@dataclass
class DatasetInfo:
    """Benchmark dataset."""

    name: str
    path: Path
    group: str
    extra_overrides: dict | None = None

    def to_dict(self):
        return {
            "name": self.name,
            "path": str(self.path),
            "group": self.group,
            "extra_overrides": self.extra_overrides,
        }


class RunTestType(Enum):
    """Run test type."""

    TORCH = "torch"
    EXPORT = "export"
    OPTIMIZE = "optimize"


class SubCommand(Enum):
    """SubCommand of benchmark."""

    TRAIN = "train"
    TEST = "test"
    EXPORT = "export"
    OPTIMIZE = "optimize"


@dataclass
class Criterion:
    """Benchmark criterion."""

    name: str
    summary: str
    compare: str
    margin: float

    def __call__(self, result_entry: pd.Series, target_entry: pd.Series) -> None:
        """Check result against given target."""
        if self.name not in result_entry or result_entry[self.name] is None or np.isnan(result_entry[self.name]):
            print(f"[Check] {self.name} not in result")
            return
        if self.name not in target_entry or target_entry[self.name] is None or np.isnan(target_entry[self.name]):
            print(f"[Check] {self.name} not in target")
            return
        if self.compare == "==":
            print(
                f"[Check] abs({self.name}:{result_entry[self.name]} - {self.name}:{target_entry[self.name]}) < {self.name}:{target_entry[self.name]} * {self.margin}",
            )
            assert abs(result_entry[self.name] - target_entry[self.name]) < target_entry[self.name] * self.margin
        elif self.compare == "<":
            print(
                f"[Check] {self.name}:{result_entry[self.name]} < {self.name}:{target_entry[self.name]} * (1.0 + {self.margin})",
            )
            assert result_entry[self.name] < target_entry[self.name] * (1.0 + self.margin)
        elif self.compare == ">":
            print(
                f"[Check] {self.name}:{result_entry[self.name]} > {self.name}:{target_entry[self.name]} * (1.0 - {self.margin})",
            )
            assert result_entry[self.name] > target_entry[self.name] * (1.0 - self.margin)

def parse_task(value: str) -> OTXTaskType:
    try:
        # Normalize input to uppercase before converting to enum.
        return OTXTaskType(value.upper())
    except ValueError:
        raise ValueError(f"'{value}' is not a valid task type. Valid options are: {', '.join([t.value for t in OTXTaskType])}")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--task",
        type=parse_task,
        choices=list(OTXTaskType),
        help="Task type to benchmark.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--num-epoch",
        type=int,
        default=200,
        help=(
            "Overrides default per-model number of epoch setting. "
            "Defaults to 0 (per-model epoch & early-stopping)."
        ),
    )
    parser.add_argument(
        "--eval-upto",
        type=str,
        default="optimize",
        choices=["train", "export", "optimize"],
        help="Choose train|export|optimize. Defaults to train.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to output summary file. Defaults to {output_root}/benchmark-summary.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        choices=["true", "false", "warn"],
        default=None,
        help="Turn on deterministic training (true/false/warn).",
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="anonymous",
        help='Sign-off the user name who launched the regression tests this time, e.g., --user-name "John Doe".',
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Previous performance test directory which contains execution results. "
            "If training was already done in a previous performance test, training is skipped and previous results are used."
        ),
    )
    parser.add_argument(
        "--test-only",
        type=str,
        choices=["all", "train", "export", "optimize"],
        default=None,
        help=(
            "Execute test only when resume argument is given. "
            "If necessary files are not found in resume directory, necessary operations can be executed. "
            "Choose all|train|export|optimize."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Which device to use.",
    )
    return parser


def current_date_str() -> str:
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    return datetime.now(tz=tz).strftime("%Y%m%d-%H%M%S")


def setup_output_root(config: Namespace, current_date: str, task: OTXTaskType) -> Path:
    output_root = config.output_root
    if output_root is None:
        # Use a temporary directory if output_root not provided
        output_root = Path(os.path.join(os.path.abspath("."), "otx-benchmark-temp"))
        output_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(output_root) / current_date / task.value
    logger.info(f"output_root = {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def get_version_tags(current_date: str) -> dict[str, str]:
    try:
        version_str = subprocess.check_output(["otx", "--version"]).decode("ascii").strip()[4:]
    except Exception:
        version_str = "unknown"
    try:
        branch_str = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
    except Exception:
        branch_str = os.environ.get("GH_CTX_REF_NAME", "unknown")
    try:
        commit_str = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    version_tags = {
        "otx_version": version_str,
        "otx_ref": commit_str,
        "test_branch": branch_str,
        "test_commit": commit_str,
        "date": current_date,
    }
    logger.info(f"version_tags = {version_tags}")
    return version_tags


def build_tags(config: Namespace, version_tags: dict[str, str]) -> dict[str, str]:
    tags = {
        **version_tags,
        "user_name": config.user_name,
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
    }
    if config.device == "gpu":
        tags["accelerator_info"] = subprocess.check_output(["nvidia-smi", "-L"]).decode().strip()
    elif config.device == "xpu":
        raw = subprocess.check_output(["xpu-smi", "discovery", "--dump", "1,2"]).decode().strip()
        tags["accelerator_info"] = "\n".join(
            [ret.replace('"', "").replace(",", " : ") for ret in raw.split("\n")[1:]]
        )
    elif config.device == "cpu":
        tags["accelerator_info"] = "cpu"
    logger.info(f"tags = {tags}")
    return tags


def load_result(result_path: Path) -> pd.DataFrame | None:
    """Load benchmark results recursively and merge as pd.DataFrame.

    Args:
        result_path (Path): Result directory or speicific file.

    Retruns:
        pd.DataFrame: Table with benchmark metrics & options
    """
    if not result_path.exists():
        return None
    # Load csv data
    csv_files = result_path.glob("**/benchmark.raw.csv") if result_path.is_dir() else [result_path]
    results = [pd.read_csv(csv_file) for csv_file in csv_files]
    if len(results) == 0:
        return None

    return pd.concat(results, ignore_index=True)


def completeness_check(
    raw_df: pd.DataFrame,
    model_list: list[ModelInfo],
    dataset_list: list[DatasetInfo],
    num_repeat: int,
) -> list[tuple[str, str, int]]:
    missing_experiments = []
    for model in model_list:
        for data in dataset_list:
            for seed in range(num_repeat):
                # query
                query = raw_df.query(
                    f"task == '{model.task}' and model == '{model.name}' and data == '{data.name}' and seed == '{seed}'"
                )
                if len(query) == 0:
                    logger.error(f"Missing data for model: {model.name} data: {data.name} seed: {seed}.")
                    missing_experiments.append((model.name, data.name, seed))

    return missing_experiments
