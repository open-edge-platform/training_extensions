# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This script is used to plot the benchmark data."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
from otx.types.task import OTXTaskType
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)



# Add parent directory to path so we can import tests module
sys.path.append(str(Path(__file__).parents[2]))

from tests.perf_v2 import (
    DATASET_COLLECTIONS,
    MODEL_COLLECTIONS,
    TASK_METRIC_MAP,
)


COMMON_CRITERIA = [
    "training:e2e_time",
    "training:epoch",
    "training:train/iter_time",
    "training:gpu_mem",
    "torch:test/{accuracy_metric}",
    "export:test/{accuracy_metric}",
    "optimize:test/{accuracy_metric}",
    "torch:test/latency",
    "export:test/latency",
    "optimize:test/latency",
    "optimize:test/e2e_time",
]


class BenchmarkPlotter:
    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = benchmark_dir

    def get_otx_version(self, xlsx_path: Path) -> str:
        """
        Get the otx version and task from the first sheet of the xlsx file.

        Args:
            xlsx_path: The path to the xlsx file.

        Returns:
            str: The otx version found in the xlsx file.
        """
        xlsx = pd.ExcelFile(xlsx_path)
        df = pd.read_excel(xlsx, sheet_name=0)
        otx_version = df["otx_version"].unique()
        if len(otx_version) > 1 :
            raise ValueError(f"Multiple otx versions found in {xlsx_path}")        
        return otx_version[0]

    def read_xlsx_sheet(self, xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
        """
        Read the sheet from the xlsx file.
        """
        xlsx = pd.ExcelFile(xlsx_path)
        try:
            return pd.read_excel(xlsx, sheet_name=sheet_name)
        except ValueError as e:
            logger.error(f"Error reading sheet {sheet_name} from {xlsx_path}: {e}")
            raise e

    def load_xlsx_benchmark_data(self, task: OTXTaskType) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load the benchmark data from the xlsx files.

        The benchmark data is a dictionary of dataframes, 
        where the keys are the otx versions and the values are the dataframes.

        The dataframes are indexed by the tasks and the test cases.

        Args:
            task: The task to load the benchmark data for.

        Returns:
            dict[str, dict[str, pd.DataFrame]]: The benchmark data with otx version as the first key,
        """
        task_dataset_dfs = {}
        for task, test_cases in DATASET_COLLECTIONS.items():
            for test_case in test_cases:
                xlsx_name = f"{task.name}-raw-{test_case.name}.xlsx"
                xlsx_paths = list(self.benchmark_dir.glob(f"**/{xlsx_name}"))
                if len(xlsx_paths) == 0:
                    msg = f"File {xlsx_name} not found"
                    raise FileNotFoundError(msg)

                logger.info(f"Found {len(xlsx_paths)} xlsx files for {task.name} {test_case.name}")
                for xlsx_path in xlsx_paths:
                    logger.info(f"Loading {xlsx_path}")
                    otx_version = self.get_otx_version(xlsx_path)
                    if otx_version not in task_dataset_dfs:
                        task_dataset_dfs[otx_version] = {}

                    sheets = {model.name: self.read_xlsx_sheet(xlsx_path, model.name) for model in MODEL_COLLECTIONS[task]}
                    task_dataset_dfs[otx_version] = sheets
        return task_dataset_dfs
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_dir", 
        type=Path, 
        help="Path to the benchmark directory",
        default=Path("tests/perf_v2/perf_history")

    )
    args = parser.parse_args()

    benchmark_plotter = BenchmarkPlotter(args.benchmark_dir)

    task_dataset_df = benchmark_plotter.load_xlsx_benchmark_data(
        OTXTaskType.MULTI_LABEL_CLS
    )
    breakpoint()

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("OTX Benchmark"),



    ]
    )