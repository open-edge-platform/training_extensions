# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX performance benchmark history summary utilities."""

from __future__ import annotations

import argparse
import fnmatch
import io
import logging
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from otx.core.types.task import OTXTaskType

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


logger = logging.getLogger(__name__)

TASK_METRIC_MAP = {
    OTXTaskType.ANOMALY_CLASSIFICATION: "f1-score",
    OTXTaskType.ANOMALY_DETECTION: "f1-score",
    OTXTaskType.ANOMALY_SEGMENTATION: "f1-score",
    OTXTaskType.MULTI_CLASS_CLS: "accuracy",
    OTXTaskType.MULTI_LABEL_CLS: "accuracy",
    OTXTaskType.H_LABEL_CLS: "accuracy",
    OTXTaskType.DETECTION: "f1-score",
    OTXTaskType.INSTANCE_SEGMENTATION: "f1-score",
    OTXTaskType.SEMANTIC_SEGMENTATION: "dice",
    OTXTaskType.VISUAL_PROMPTING: "dice",
    OTXTaskType.KEYPOINT_DETECTION: "accuracy",
}


METADATA_ENTRIES = [
    "date",
    "task",
    "model",
    "data_group",
    "data",
    "otx_version",
    "otx_ref",
    "test_branch",
    "test_commit",
    "cpu_info",
    "accelerator_info",
    "user_name",
    "machine_name",
]


def load(root_dir: Path, pattern="*raw*.csv") -> pd.DataFrame:
    """Load all csv files and csv in zip files."""

    history = []
    # Load csv files in the directory
    csv_files = root_dir.rglob(pattern)
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        history.append(data)

    # Load csv files in zip files
    zip_files = Path(root_dir).rglob("*.zip")
    for zip_file in zip_files:
        with ZipFile(zip_file) as zf:
            csv_files = fnmatch.filter(zf.namelist(), pattern)
            for csv_file in csv_files:
                csv_bytes = io.BytesIO(zf.read(csv_file))
                data = pd.read_csv(csv_bytes)
                history.append(data)
    if len(history) == 0:
        return pd.DataFrame()
    history = pd.concat(history, ignore_index=True)
    # Post process
    version_entry = "otx_version" if "otx_version" in history else "version"
    history[version_entry] = history[version_entry].astype(str)
    history["seed"] = history["seed"].fillna(0)
    history = average(
        history,
        [version_entry, "task", "model", "data_group", "data", "seed"],
    )  # Average mulitple retrials w/ same seed
    if "index" in history:
        history.drop("index", axis=1)
    return history


def average(raw_data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average raw data w.r.t. given keys."""
    if raw_data is None or len(raw_data) == 0:
        return pd.DataFrame()
    # To avoid SettingWithCopyWarning
    raw_data = raw_data.copy()
    # Preproc
    for col in METADATA_ENTRIES:
        raw_data.loc[:, col] = raw_data[col].astype(str)  # Prevent strings like '2.0.0' being loaded as float
    # Average by keys
    grouped = raw_data.groupby(keys)
    aggregated = grouped.mean(numeric_only=True)
    # Merge tag columns (non-numeric & non-index)
    tag_columns = set(raw_data.columns) - set(aggregated.columns) - set(keys)
    for col in tag_columns:
        # Take common string prefix such as: ["data/1", "data/2", "data/3"] -> "data/"
        aggregated[col] = grouped[col].agg(lambda x: os.path.commonprefix(x.tolist()))
    return aggregated.reset_index()


def aggregate(raw_data: pd.DataFrame, metrics: list[str]) -> list[pd.DataFrame]:
    """Summarize raw data into pivot table w.r.t given metrics"""
    if raw_data is None or len(raw_data) == 0:
        return [pd.DataFrame()]

    # Drop `seed` column if it is raw_data as we don't need to average it
    if "seed" in raw_data.columns:
        raw_data = raw_data.drop(columns=["seed"])

    for col in METADATA_ENTRIES:
        raw_data.loc[:, col] = raw_data[col].astype(str)  # Prevent strings like '2.0.0' being loaded as float

    grouped_data = raw_data.groupby(
        [
            "otx_version",
            "task",
            "model",
            "data",
            "test_branch",
            "test_commit",
            "data_group",
        ],
    )
    aggregated = grouped_data.agg({metric: ["mean", "std"] for metric in metrics}).reset_index()

    # Flatten the MultiIndex columns, excluding 'otx_version', 'task', 'model', 'data_group'
    cols_to_exclude = {"otx_version", "task", "model", "data"}
    aggregated.columns = [
        ("_".join(col) if col[0] not in cols_to_exclude else col[0]) for col in aggregated.columns.to_numpy()
    ]

    # Get metrics (int/float) columns
    number_cols = aggregated.select_dtypes(include=["number"]).columns.to_list()

    # Get metadata (str type) columns
    meta_cols = aggregated.select_dtypes(include=["object"]).columns.to_list()
    if "model" in meta_cols:
        meta_cols.remove("model")

    rearrange_cols = [
        "model",
        *number_cols,
        *meta_cols,
    ]

    # Rearrange columns
    aggregated = aggregated.reindex(columns=rearrange_cols)

    # Individualize each sheet by dataset
    dataset_dfs = []
    datase_names = aggregated["data"].unique()
    for data_name in datase_names:
        data = aggregated[aggregated["data"] == data_name]
        dataset_dfs.append(data)
    return dataset_dfs


def summarize_table(history: pd.DataFrame, task: OTXTaskType) -> list[pd.DataFrame]:
    """Summarize benchmark histoy table by task."""
    score_metric = TASK_METRIC_MAP[task]

    # Metrics to summarize in aggregated table
    metrics = [
        "training:e2e_time",
        "training:epoch",
        "training:train/iter_time",
        "training:gpu_mem",
        f"torch:test/{score_metric}",
        f"export:test/{score_metric}",
        f"optimize:test/{score_metric}",
        "torch:test/latency",
        "export:test/latency",
        "optimize:test/latency",
        "optimize:test/e2e_time",
    ]

    raw_task_data = history.query(f"task == '{task.value}'")
    dataset_dfs = aggregate(raw_task_data, metrics)

    # Round all numeric columns to 4 decimal places
    for df in dataset_dfs:
        columns = df.select_dtypes(include=["number"]).columns
        for col in columns:
            df[col] = df[col].round(4)
    return dataset_dfs


def create_raw_dataset_xlsx(
    raw_data: pd.DataFrame,
    task: OTXTaskType,
    output_root: Path,
):
    """Create raw_values_<dataset>.xlsx file for each dataset.

    Args:
        raw_data (pd.DataFrame): _description_
        task (str): _description_
        output_root (Path): _description_
    """
    from tests.perf_v2 import CRITERIA_COLLECTIONS

    col_names = ["seed"]
    col_names.extend([criterion.name for criterion in CRITERIA_COLLECTIONS[task]])
    col_names.extend(METADATA_ENTRIES)

    raw_data_task = raw_data.query(f"task == '{task.value}'")
    for dataset in raw_data_task["data"].unique():
        raw_data_dataset = raw_data_task.query(f"data == '{dataset}'")
        with pd.ExcelWriter(output_root / f"{task.value}-raw-{dataset}.xlsx") as writer:
            for model in raw_data_dataset["model"].unique():
                raw_data_model_df = raw_data_dataset.query(f"model == '{model}'")
                raw_data_model_df = raw_data_model_df.reindex(columns=col_names)
                raw_data_model_df = raw_data_model_df.dropna(axis=1)
                raw_data_model_df.to_excel(writer, sheet_name=model, index=False)
        logger.info(f"    Saved {task.value} raw data to {output_root / f'{task.value}-raw-{dataset}.xlsx'!s}")


def summarize_task(raw_data: pd.DataFrame, task: OTXTaskType, output_root: Path):
    """Process and save task-specific data and summaries."""
    # Create raw_values_<dataset>.xlsx file for each dataset
    create_raw_dataset_xlsx(raw_data, task, output_root)

    # Use summarize_table function to get a detailed summary for each task
    dataset_dfs = summarize_table(raw_data, task)

    # Save the detailed summary data to an Excel file, including the index
    task_str = task.replace("/", "_")

    # Create a summary Excel file for each task: <task>-aggregated.xlsx
    aggregate_xlsx_path = output_root / f"{task_str}-aggregated.xlsx"
    with pd.ExcelWriter(aggregate_xlsx_path) as writer:
        for dataset_df in dataset_dfs:
            dataset_df.to_excel(writer, sheet_name=dataset_df["data"].iloc[0], index=False)
    logger.info(f"    Saved {task.value} summary to {aggregate_xlsx_path.resolve()!s}")


def task_high_level_summary(raw_data: pd.DataFrame, task: OTXTaskType, output_root: Path):
    """Summarize high-level task performance over all datasets, with one row per model."""

    raw_task_data = raw_data.query(f"task == '{task.value}'")
    if raw_task_data is None or len(raw_task_data) == 0:
        msg = f"No data found for task {task.value}"
        raise ValueError(msg)

    # Drop `seed` column if it is raw_data as we don't need to average it
    if "seed" in raw_task_data.columns:
        raw_task_data = raw_task_data.drop(columns=["seed"])

    for col in METADATA_ENTRIES:
        raw_task_data.loc[:, col] = raw_task_data[col].astype(str)  # Prevent strings like '2.0.0' being loaded as float

    metrics = raw_task_data.select_dtypes(include=["number"]).columns.to_list()

    # Group by model instead of just otx_version and task
    grouped_data = raw_task_data.groupby(["otx_version", "task", "model"])
    aggregated = grouped_data.agg({metric: ["mean", "std"] for metric in metrics}).reset_index()

    # Flatten the MultiIndex columns
    cols_to_exclude = {"otx_version", "task", "model", "data"}
    aggregated.columns = [
        ("_".join(col) if col[0] not in cols_to_exclude else col[0]) for col in aggregated.columns.to_numpy()
    ]

    number_cols = aggregated.select_dtypes(include=["number"]).columns.to_list()
    meta_cols = aggregated.select_dtypes(include=["object"]).columns.to_list()
    for col in meta_cols:
        if col in ["model", "task", "otx_version"]:
            meta_cols.remove(col)

    # Rearrange columns to match the order in aggregate function
    aggregated = aggregated.reindex(
        columns=[
            "otx_version",
            "task",
            "model",
            *number_cols,
            *meta_cols,
        ],
    )

    # Round all numeric columns to 4 decimal places
    columns = aggregated.select_dtypes(include=["number"]).columns
    for col in columns:
        aggregated[col] = aggregated[col].round(4)

    # Save the high-level summary data to an Excel file
    task_high_level_summary_xlsx_path = output_root / f"{task.value}-high-level-summary.xlsx"
    aggregated.to_excel(task_high_level_summary_xlsx_path, index=False)
    logger.info(f"    Saved {task.value} high-level summary to {task_high_level_summary_xlsx_path.resolve()!s}")


if __name__ == "__main__":
    """Load csv files in directory & zip files, merge them, summarize per task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    parser.add_argument("--pattern", default="*raw*.csv")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    logger.info(f"Loading {args.pattern} in {input_root}...")
    raw_data = load(input_root, pattern=args.pattern)
    if len(raw_data) == 0:
        logger.error("No data loaded")
        sys.exit(-1)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(output_root / "perf-benchmark-raw-all.csv", index=False)
    logger.info(f"Saved merged raw data to {output_root / 'perf-benchmark-raw-all.csv'}")

    # Get task-level performance benchmark
    tasks = sorted(raw_data["task"].unique())
    for task in tasks:
        summarize_task(raw_data, OTXTaskType[task], output_root)
        task_high_level_summary(raw_data, OTXTaskType[task], output_root)
