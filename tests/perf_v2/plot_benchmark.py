# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This script is used to plot the benchmark data using Plotly Dash."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from otx.types.task import OTXTaskType

# Add parent directory to path so we can import tests module
sys.path.append(str(Path(__file__).parents[2]))

from tests.perf_v2 import (  # noqa: E402
    DATASET_COLLECTIONS,
    TASK_METRIC_MAP,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the task constants as requested
TASK_CONSTANTS = {
    "ANOMALY": OTXTaskType.ANOMALY,
    "MULTI_CLASS_CLS": OTXTaskType.MULTI_CLASS_CLS,
    "MULTI_LABEL_CLS": OTXTaskType.MULTI_LABEL_CLS,
    "H_LABEL_CLS": OTXTaskType.H_LABEL_CLS,
    "DETECTION": OTXTaskType.DETECTION,
    "ROTATED_DETECTION": OTXTaskType.ROTATED_DETECTION,
    "KEYPOINT_DETECTION": OTXTaskType.KEYPOINT_DETECTION,
    "INSTANCE_SEGMENTATION": OTXTaskType.INSTANCE_SEGMENTATION,
    "SEMANTIC_SEGMENTATION": OTXTaskType.SEMANTIC_SEGMENTATION,
}

AVAILABLE_TASKS = [
    OTXTaskType.ANOMALY,
    OTXTaskType.MULTI_CLASS_CLS,
    OTXTaskType.MULTI_LABEL_CLS,
    OTXTaskType.H_LABEL_CLS,
    OTXTaskType.DETECTION,
    OTXTaskType.KEYPOINT_DETECTION,
    OTXTaskType.INSTANCE_SEGMENTATION,
    OTXTaskType.SEMANTIC_SEGMENTATION,
]


class BenchmarkDashboard:
    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = benchmark_dir
        self.app = Dash(__name__)
        self.benchmark_data: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        self.available_versions: list[str] = []
        self.setup_layout()
        self.setup_callbacks()

    def get_available_versions(self) -> list[str]:
        """Get all available OTX versions from the benchmark directory."""
        version_dirs = [d.name for d in self.benchmark_dir.iterdir() if d.is_dir()]
        return sorted(version_dirs)

    def get_available_tasks_for_version(self, version: str) -> list[OTXTaskType]:
        """Get available tasks for a specific version."""
        version_dir = self.benchmark_dir / version
        if not version_dir.exists():
            return []

        task_dirs = [d.name.upper() for d in version_dir.iterdir() if d.is_dir()]
        return [task for task in AVAILABLE_TASKS if task.value.upper() in task_dirs]

    def load_aggregated_data(self, task: OTXTaskType, version: str) -> dict[str, pd.DataFrame] | None:
        """Load aggregated benchmark data for a specific task and version."""
        xlsx_name = f"{task.value}-aggregated.xlsx"
        xlsx_path = self.benchmark_dir / version / task.value.lower() / xlsx_name

        if not xlsx_path.exists():
            logger.warning(f"Aggregated file not found: {xlsx_path}")
            return None

        logger.info(f"Loading {xlsx_path}")
        data_frames = {}

        if task in DATASET_COLLECTIONS:
            for dataset in DATASET_COLLECTIONS[task]:
                try:
                    data_frame = pd.read_excel(xlsx_path, sheet_name=dataset.name)
                    data_frames[dataset.name] = data_frame
                except Exception as e:  # noqa: PERF203
                    logger.warning(f"Could not load sheet {dataset.name} from {xlsx_path}: {e}")

        return data_frames if data_frames else None

    def load_all_benchmark_data(self):
        """Load all available benchmark data."""
        self.available_versions = self.get_available_versions()
        logger.info(f"Found versions: {self.available_versions}")

        for version in self.available_versions:
            self.benchmark_data[version] = {}
            available_tasks = self.get_available_tasks_for_version(version)

            for task in available_tasks:
                data = self.load_aggregated_data(task, version)
                if data:
                    self.benchmark_data[version][task.value] = data

    def create_metric_comparison_plot(self, task: OTXTaskType, dataset_name: str, metric_type: str = "accuracy"):
        """Create a plot comparing metrics across versions for a specific task and dataset."""
        if not self.benchmark_data:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Determine the metric column name based on task
        if metric_type == "accuracy":
            if task in TASK_METRIC_MAP:
                metric_col = f"torch:test/{TASK_METRIC_MAP[task]}_mean"
            else:
                metric_col = "torch:test/accuracy_mean"
        else:  # latency
            metric_col = "torch:test/latency_mean"

        # Collect data across versions
        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if metric_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return go.Figure().add_annotation(
                text=f"No {metric_type} data available for {dataset_name}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - {dataset_name} - {metric_type.title()} Comparison",
            barmode="group",
            height=500,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title=f"{metric_type.title()}",
            legend_title="OTX Version",
            showlegend=True,
        )

        return fig

    def create_accuracy_comparison_plots(self, task: OTXTaskType, dataset_name: str):
        """Create multiple accuracy comparison plots for different test stages."""
        if not self.benchmark_data:
            return html.Div("No data available")

        # Define the accuracy metrics for different stages
        accuracy_metric = TASK_METRIC_MAP.get(task, "accuracy")
        stages = [
            ("torch", f"torch:test/{accuracy_metric}_mean", "Torch Model"),
            ("export", f"export:test/{accuracy_metric}_mean", "Exported Model"),
            ("optimize", f"optimize:test/{accuracy_metric}_mean", "Optimized Model"),
        ]

        plots = []

        for _, metric_col, stage_title in stages:
            # Collect data across versions for this stage
            plot_data = []

            for version in self.available_versions:
                if (
                    version in self.benchmark_data
                    and task.value in self.benchmark_data[version]
                    and dataset_name in self.benchmark_data[version][task.value]
                ):
                    data_frame = self.benchmark_data[version][task.value][dataset_name]

                    if metric_col in data_frame.columns:
                        for _, row in data_frame.iterrows():
                            if not pd.isna(row[metric_col]):
                                plot_data.append(
                                    {
                                        "Version": version,
                                        "Model": row.get("model", "Unknown"),
                                        "Metric": row[metric_col],
                                        "Dataset": dataset_name,
                                        "Stage": stage_title,
                                    },
                                )

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create the plot
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="Metric",
                    color="Version",
                    title=f"{task.value} - {dataset_name} - {stage_title} Accuracy",
                    barmode="group",
                    height=400,
                )

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title=f"{accuracy_metric.title()}",
                    legend_title="OTX Version",
                    showlegend=True,
                    margin={"t": 50, "b": 50, "l": 50, "r": 50},
                )

                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Accuracy Comparison",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            dcc.Graph(figure=fig),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )
            else:
                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Accuracy Comparison",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            html.P(
                                f"No {stage_title.lower()} accuracy data available for {dataset_name}",
                                style={
                                    "textAlign": "center",
                                    "color": "#7f8c8d",
                                    "fontStyle": "italic",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )

        return html.Div(plots)

    def create_latency_comparison_plots(self, task: OTXTaskType, dataset_name: str):
        """Create multiple latency comparison plots for different test stages."""
        if not self.benchmark_data:
            return html.Div("No data available")

        # Define the latency metrics for different stages
        stages = [
            ("torch", "torch:test/latency_mean", "Torch Model"),
            ("export", "export:test/latency_mean", "Exported Model"),
            ("optimize", "optimize:test/latency_mean", "Optimized Model"),
        ]

        plots = []

        for _, metric_col, stage_title in stages:
            # Collect data across versions for this stage
            plot_data = []

            for version in self.available_versions:
                if (
                    version in self.benchmark_data
                    and task.value in self.benchmark_data[version]
                    and dataset_name in self.benchmark_data[version][task.value]
                ):
                    data_frame = self.benchmark_data[version][task.value][dataset_name]

                    if metric_col in data_frame.columns:
                        for _, row in data_frame.iterrows():
                            if not pd.isna(row[metric_col]):
                                plot_data.append(
                                    {
                                        "Version": version,
                                        "Model": row.get("model", "Unknown"),
                                        "Metric": row[metric_col],
                                        "Dataset": dataset_name,
                                        "Stage": stage_title,
                                    },
                                )

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create the plot
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="Metric",
                    color="Version",
                    title=f"{task.value} - {dataset_name} - {stage_title} Latency",
                    barmode="group",
                    height=400,
                )

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Latency (ms)",
                    legend_title="OTX Version",
                    showlegend=True,
                    margin={"t": 50, "b": 50, "l": 50, "r": 50},
                )

                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Latency Comparison",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            dcc.Graph(figure=fig),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )
            else:
                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Latency Comparison",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            html.P(
                                f"No {stage_title.lower()} latency data available for {dataset_name}",
                                style={
                                    "textAlign": "center",
                                    "color": "#7f8c8d",
                                    "fontStyle": "italic",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )

        return html.Div(plots)

    def create_training_time_comparison_plot(self, task: OTXTaskType, dataset_name: str):
        """Create training time comparison plot."""
        if not self.benchmark_data:
            return html.Div("No data available")

        metric_col = "training:e2e_time_mean"
        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if metric_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return html.Div(
                [
                    html.H4(
                        "Training Time Comparison",
                        style={
                            "textAlign": "center",
                            "marginTop": "30px",
                            "marginBottom": "10px",
                            "color": "#2c3e50",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.P(
                        f"No training time data available for {dataset_name}",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#fafafa",
                },
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - {dataset_name} - Training Time Comparison",
            barmode="group",
            height=400,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Training Time (seconds)",
            legend_title="OTX Version",
            showlegend=True,
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
        )

        return html.Div(
            [
                html.H4(
                    "Training Time Comparison",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "10px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                dcc.Graph(figure=fig),
            ],
            style={
                "marginBottom": "20px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "5px",
                "padding": "15px",
                "backgroundColor": "#fafafa",
            },
        )

    def create_epoch_comparison_plot(self, task: OTXTaskType, dataset_name: str) -> html.Div:
        """Create epoch comparison plot."""
        if not self.benchmark_data:
            return html.Div("No data available")

        metric_col = "training:epoch_mean"
        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if metric_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return html.Div(
                [
                    html.H4(
                        "Epoch Comparison",
                        style={
                            "textAlign": "center",
                            "marginTop": "30px",
                            "marginBottom": "10px",
                            "color": "#2c3e50",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.P(
                        f"No epoch data available for {dataset_name}",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#fafafa",
                },
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - {dataset_name} - Epoch Comparison",
            barmode="group",
            height=400,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Number of Epochs",
            legend_title="OTX Version",
            showlegend=True,
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
        )

        return html.Div(
            [
                html.H4(
                    "Epoch Comparison",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "10px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                dcc.Graph(figure=fig),
            ],
            style={
                "marginBottom": "20px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "5px",
                "padding": "15px",
                "backgroundColor": "#fafafa",
            },
        )

    def create_iter_time_comparison_plot(self, task: OTXTaskType, dataset_name: str) -> html.Div:
        """Create iteration time comparison plot."""
        if not self.benchmark_data:
            return html.Div("No data available")

        metric_col = "training:train/iter_time_mean"
        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if metric_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return html.Div(
                [
                    html.H4(
                        "Iteration Time Comparison",
                        style={
                            "textAlign": "center",
                            "marginTop": "30px",
                            "marginBottom": "10px",
                            "color": "#2c3e50",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.P(
                        f"No iteration time data available for {dataset_name}",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#fafafa",
                },
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - {dataset_name} - Iteration Time Comparison",
            barmode="group",
            height=400,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Iteration Time (seconds)",
            legend_title="OTX Version",
            showlegend=True,
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
        )

        return html.Div(
            [
                html.H4(
                    "Iteration Time Comparison",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "10px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                dcc.Graph(figure=fig),
            ],
            style={
                "marginBottom": "20px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "5px",
                "padding": "15px",
                "backgroundColor": "#fafafa",
            },
        )

    def create_latency_accuracy_scatter(self, task: OTXTaskType, dataset_name: str) -> go.Figure:
        """Create a scatter plot showing accuracy vs latency for different versions."""
        if not self.benchmark_data:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Determine metric columns
        accuracy_col = f"torch:test/{TASK_METRIC_MAP.get(task, 'accuracy')}_mean"
        latency_col = "torch:test/latency_mean"

        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if accuracy_col in data_frame.columns and latency_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[accuracy_col]) and not pd.isna(row[latency_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Accuracy": row[accuracy_col],
                                    "Latency": row[latency_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return go.Figure().add_annotation(
                text=f"No accuracy vs latency data available for {dataset_name}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        plot_df = pd.DataFrame(plot_data)

        fig = px.scatter(
            plot_df,
            x="Latency",
            y="Accuracy",
            color="Version",
            symbol="Model",
            title=f"{task.value} - {dataset_name} - Accuracy vs Latency",
            height=500,
            hover_data=["Model"],
        )

        fig.update_layout(
            xaxis_title="Latency (ms)",
            yaxis_title="Accuracy",
            legend_title="OTX Version",
        )

        return fig

    def create_averaged_data(self, task: OTXTaskType) -> dict[str, pd.DataFrame]:
        """Create averaged data across all datasets for a task."""
        if not self.benchmark_data:
            return {}

        averaged_data = {}

        for version in self.available_versions:
            if version in self.benchmark_data and task.value in self.benchmark_data[version]:
                version_data = self.benchmark_data[version][task.value]
                all_datasets_data = []

                # Collect all data from all datasets for this version
                for dataset_name, df in version_data.items():
                    if not df.empty:
                        df_copy = df.copy()
                        df_copy["dataset"] = dataset_name
                        all_datasets_data.append(df_copy)

                if all_datasets_data:
                    # Combine all datasets
                    combined_df = pd.concat(all_datasets_data, ignore_index=True)

                    # Group by model and calculate mean across datasets
                    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
                    grouped = combined_df.groupby("model")[numeric_columns].mean().reset_index()

                    # Keep other important columns
                    for col in ["otx_version", "task"]:
                        if col in combined_df.columns:
                            grouped[col] = combined_df[col].iloc[0]

                    averaged_data[version] = grouped

        return averaged_data

    def create_averaged_plots(self, task: OTXTaskType, metric_type: str) -> html.Div | dcc.Graph:
        """Create plots using averaged data across all datasets for a task."""
        if metric_type == "accuracy":
            return self.create_averaged_accuracy_comparison_plots(task)
        if metric_type == "latency":
            return self.create_averaged_latency_comparison_plots(task)
        if metric_type == "training_time":
            return self.create_averaged_training_metric_plot(
                task,
                "training:e2e_time_mean",
                "Training Time",
                "Training Time (seconds)",
            )
        if metric_type == "epoch":
            return self.create_averaged_training_metric_plot(
                task,
                "training:epoch_mean",
                "Epoch",
                "Number of Epochs",
            )
        if metric_type == "iter_time":
            return self.create_averaged_training_metric_plot(
                task,
                "training:train/iter_time_mean",
                "Iteration Time",
                "Iteration Time (seconds)",
            )
        if metric_type == "gpu_mem":
            return self.create_averaged_training_metric_plot(
                task,
                "training:gpu_mem_mean",
                "GPU Memory",
                "GPU Memory (GB)",
            )
        if metric_type == "scatter":
            fig = self.create_averaged_latency_accuracy_scatter(task)
            return dcc.Graph(figure=fig)
        return html.Div("Invalid metric type selected")

    def create_averaged_accuracy_comparison_plots(self, task: OTXTaskType) -> html.Div:
        """Create accuracy comparison plots using averaged data across datasets."""
        averaged_data = self.create_averaged_data(task)

        if not averaged_data:
            return html.Div("No data available for averaging")

        # Define the accuracy metrics for different stages
        accuracy_metric = TASK_METRIC_MAP.get(task, "accuracy")
        stages = [
            ("torch", f"torch:test/{accuracy_metric}_mean", "Torch Model"),
            ("export", f"export:test/{accuracy_metric}_mean", "Exported Model"),
            ("optimize", f"optimize:test/{accuracy_metric}_mean", "Optimized Model"),
        ]

        plots = []

        for _, metric_col, stage_title in stages:
            # Collect averaged data across versions for this stage
            plot_data = []

            for version, df in averaged_data.items():
                if metric_col in df.columns:
                    for _, row in df.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Stage": stage_title,
                                },
                            )

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create the plot
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="Metric",
                    color="Version",
                    title=f"{task.value} - Average Across Datasets - {stage_title} Accuracy",
                    barmode="group",
                    height=400,
                )

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title=f"{accuracy_metric.title()}",
                    legend_title="OTX Version",
                    showlegend=True,
                    margin={"t": 50, "b": 50, "l": 50, "r": 50},
                )

                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Accuracy Comparison (Averaged)",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            dcc.Graph(figure=fig),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )
            else:
                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Accuracy Comparison (Averaged)",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            html.P(
                                f"No averaged {stage_title.lower()} accuracy data available",
                                style={
                                    "textAlign": "center",
                                    "color": "#7f8c8d",
                                    "fontStyle": "italic",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )

        return html.Div(plots)

    def create_averaged_latency_comparison_plots(self, task: OTXTaskType) -> html.Div:
        """Create latency comparison plots using averaged data across datasets."""
        averaged_data = self.create_averaged_data(task)

        if not averaged_data:
            return html.Div("No data available for averaging")

        # Define the latency metrics for different stages
        stages = [
            ("torch", "torch:test/latency_mean", "Torch Model"),
            ("export", "export:test/latency_mean", "Exported Model"),
            ("optimize", "optimize:test/latency_mean", "Optimized Model"),
        ]

        plots = []

        for _, metric_col, stage_title in stages:
            # Collect averaged data across versions for this stage
            plot_data = []

            for version, df in averaged_data.items():
                if metric_col in df.columns:
                    for _, row in df.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Stage": stage_title,
                                },
                            )

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create the plot
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="Metric",
                    color="Version",
                    title=f"{task.value} - Average Across Datasets - {stage_title} Latency",
                    barmode="group",
                    height=400,
                )

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Latency (ms)",
                    legend_title="OTX Version",
                    showlegend=True,
                    margin={"t": 50, "b": 50, "l": 50, "r": 50},
                )

                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Latency Comparison (Averaged)",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            dcc.Graph(figure=fig),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )
            else:
                plots.append(
                    html.Div(
                        [
                            html.H4(
                                f"{stage_title} Latency Comparison (Averaged)",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "30px",
                                    "marginBottom": "10px",
                                    "color": "#2c3e50",
                                    "fontFamily": "Arial, sans-serif",
                                },
                            ),
                            html.P(
                                f"No averaged {stage_title.lower()} latency data available",
                                style={
                                    "textAlign": "center",
                                    "color": "#7f8c8d",
                                    "fontStyle": "italic",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "5px",
                            "padding": "15px",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                )

        return html.Div(plots)

    def create_averaged_training_metric_plot(
        self,
        task: OTXTaskType,
        metric_col: str,
        metric_name: str,
        y_label: str,
    ) -> html.Div:
        """Create training metric comparison plot using averaged data across datasets."""
        averaged_data = self.create_averaged_data(task)

        if not averaged_data:
            return html.Div("No data available for averaging")

        plot_data = []

        for version, df in averaged_data.items():
            if metric_col in df.columns:
                for _, row in df.iterrows():
                    if not pd.isna(row[metric_col]):
                        plot_data.append(
                            {
                                "Version": version,
                                "Model": row.get("model", "Unknown"),
                                "Metric": row[metric_col],
                            },
                        )

        if not plot_data:
            return html.Div(
                [
                    html.H4(
                        f"{metric_name} Comparison (Averaged)",
                        style={
                            "textAlign": "center",
                            "marginTop": "30px",
                            "marginBottom": "10px",
                            "color": "#2c3e50",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.P(
                        f"No averaged {metric_name.lower()} data available",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#fafafa",
                },
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - Average Across Datasets - {metric_name}",
            barmode="group",
            height=400,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title=y_label,
            legend_title="OTX Version",
            showlegend=True,
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
        )

        return html.Div(
            [
                html.H4(
                    f"{metric_name} Comparison (Averaged)",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "10px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                dcc.Graph(figure=fig),
            ],
            style={
                "marginBottom": "20px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "5px",
                "padding": "15px",
                "backgroundColor": "#fafafa",
            },
        )

    def create_averaged_latency_accuracy_scatter(self, task: OTXTaskType) -> go.Figure:
        """Create scatter plot using averaged data across datasets."""
        averaged_data = self.create_averaged_data(task)

        if not averaged_data:
            return go.Figure().add_annotation(
                text="No data available for averaging",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Determine metric columns
        accuracy_col = f"torch:test/{TASK_METRIC_MAP.get(task, 'accuracy')}_mean"
        latency_col = "torch:test/latency_mean"

        plot_data = []

        for version, df in averaged_data.items():
            if accuracy_col in df.columns and latency_col in df.columns:
                for _, row in df.iterrows():
                    if not pd.isna(row[accuracy_col]) and not pd.isna(row[latency_col]):
                        plot_data.append(
                            {
                                "Version": version,
                                "Model": row.get("model", "Unknown"),
                                "Accuracy": row[accuracy_col],
                                "Latency": row[latency_col],
                            },
                        )

        if not plot_data:
            return go.Figure().add_annotation(
                text="No averaged accuracy vs latency data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        plot_df = pd.DataFrame(plot_data)

        fig = px.scatter(
            plot_df,
            x="Latency",
            y="Accuracy",
            color="Version",
            symbol="Model",
            title=f"{task.value} - Average Across Datasets - Accuracy vs Latency",
            height=500,
            hover_data=["Model"],
        )

        fig.update_layout(
            xaxis_title="Latency (ms)",
            yaxis_title="Accuracy",
            legend_title="OTX Version",
        )

        return fig

    def create_gpu_memory_comparison_plot(self, task: OTXTaskType, dataset_name: str) -> html.Div:
        """Create GPU memory comparison plot."""
        if not self.benchmark_data:
            return html.Div("No data available")

        metric_col = "training:gpu_mem_mean"
        plot_data = []

        for version in self.available_versions:
            if (
                version in self.benchmark_data
                and task.value in self.benchmark_data[version]
                and dataset_name in self.benchmark_data[version][task.value]
            ):
                data_frame = self.benchmark_data[version][task.value][dataset_name]

                if metric_col in data_frame.columns:
                    for _, row in data_frame.iterrows():
                        if not pd.isna(row[metric_col]):
                            plot_data.append(
                                {
                                    "Version": version,
                                    "Model": row.get("model", "Unknown"),
                                    "Metric": row[metric_col],
                                    "Dataset": dataset_name,
                                },
                            )

        if not plot_data:
            return html.Div(
                [
                    html.H4(
                        "GPU Memory Comparison",
                        style={
                            "textAlign": "center",
                            "marginTop": "30px",
                            "marginBottom": "10px",
                            "color": "#2c3e50",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.P(
                        f"No GPU memory data available for {dataset_name}",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#fafafa",
                },
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        fig = px.bar(
            plot_df,
            x="Model",
            y="Metric",
            color="Version",
            title=f"{task.value} - {dataset_name} - GPU Memory Comparison",
            barmode="group",
            height=400,
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="GPU Memory (GB)",
            legend_title="OTX Version",
            showlegend=True,
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
        )

        return html.Div(
            [
                html.H4(
                    "GPU Memory Comparison",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "10px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                dcc.Graph(figure=fig),
            ],
            style={
                "marginBottom": "20px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "5px",
                "padding": "15px",
                "backgroundColor": "#fafafa",
            },
        )

    def get_datasets_for_task(self, task: OTXTaskType) -> list[str]:
        """Get available datasets for a specific task."""
        if task not in DATASET_COLLECTIONS:
            return []
        return [dataset.name for dataset in DATASET_COLLECTIONS[task]]

    def setup_layout(self) -> None:
        """Setup the Dash app layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "OTX Benchmark Dashboard",
                    style={
                        "textAlign": "center",
                        "marginBottom": "30px",
                        "color": "#2c3e50",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Refresh Data",
                            id="refresh-button",
                            style={
                                "marginBottom": "20px",
                                "padding": "10px 20px",
                                "backgroundColor": "#3498db",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontSize": "14px",
                            },
                        ),
                        html.Div(id="data-status", style={"marginBottom": "20px"}),
                    ],
                    style={"textAlign": "center"},
                ),
                dcc.Tabs(
                    id="task-tabs",
                    children=[
                        dcc.Tab(
                            label=task_name.replace("_", " ").title(),
                            value=task.value,
                            id=f"task-tab-{task.value}",
                        )
                        for task_name, task in TASK_CONSTANTS.items()
                        if task in AVAILABLE_TASKS
                    ],
                    style={
                        "marginBottom": "20px",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                html.Div(id="dataset-tabs-container"),
                html.Div(id="plots-container", style={"marginTop": "20px"}),
            ],
            style={
                "maxWidth": "1200px",
                "margin": "0 auto",
                "padding": "20px",
                "fontFamily": "Arial, sans-serif",
            },
        )

    def setup_callbacks(self) -> None:
        """Setup Dash callbacks."""

        @self.app.callback(
            Output("data-status", "children"),
            Input("refresh-button", "n_clicks"),
        )
        def refresh_data(n_clicks: int | None) -> html.Div:
            self.load_all_benchmark_data()
            return html.Div(
                [
                    html.P(f"Data loaded for versions: {', '.join(self.available_versions)}"),
                    html.P(
                        f"Available tasks: {len([task for version_data in self.benchmark_data.values() for task in version_data])} task-version combinations",
                    ),
                ],
                style={"color": "green"},
            )

        @self.app.callback(
            Output("dataset-tabs-container", "children"),
            Input("task-tabs", "value"),
        )
        def update_dataset_tabs(selected_task: str) -> html.Div | dcc.Tabs:
            if not selected_task:
                return html.Div("Please select a task")

            try:
                task = OTXTaskType(selected_task)
                datasets = self.get_datasets_for_task(task)

                if not datasets:
                    return html.Div(f"No datasets available for {selected_task}")

                tabs = [
                    dcc.Tab(
                        label=dataset.replace("_", " ").title(),
                        value=dataset,
                        id=f"dataset-tab-{dataset}",
                    )
                    for dataset in datasets
                ]

                # Add Average tab
                tabs.append(
                    dcc.Tab(
                        label="Average",
                        value="average",
                        id="dataset-tab-average",
                    ),
                )

                return dcc.Tabs(
                    id="dataset-tabs",
                    children=tabs,
                    style={"marginTop": "10px"},
                )
            except ValueError:
                return html.Div(f"Invalid task: {selected_task}")

        @self.app.callback(
            Output("plots-container", "children"),
            [Input("dataset-tabs", "value"), Input("task-tabs", "value")],
        )
        def update_plots(selected_dataset: str, selected_task: str) -> html.Div:
            if not selected_dataset or not selected_task:
                return html.Div("Please select both a task and dataset")

            try:
                task = OTXTaskType(selected_task)

                # Create metric selection buttons
                metric_buttons = html.Div(
                    [
                        html.H3(
                            f"Metrics for {selected_dataset.replace('_', ' ').title() if selected_dataset != 'average' else 'Average Across All Datasets'}",
                            style={
                                "marginTop": "20px",
                                "marginBottom": "15px",
                                "color": "#2c3e50",
                                "fontFamily": "Arial, sans-serif",
                            },
                        ),
                        dcc.RadioItems(
                            id="metric-selector",
                            options=[
                                {"label": "Accuracy Comparison", "value": "accuracy"},
                                {"label": "Latency Comparison", "value": "latency"},
                                {"label": "Training Time Comparison", "value": "training_time"},
                                {"label": "Epoch Comparison", "value": "epoch"},
                                {"label": "Iteration Time Comparison", "value": "iter_time"},
                                {"label": "GPU Memory Comparison", "value": "gpu_mem"},
                                {"label": "Accuracy vs Latency", "value": "scatter"},
                            ],
                            value="accuracy",
                            style={
                                "marginBottom": "20px",
                                "fontSize": "16px",
                                "fontFamily": "Arial, sans-serif",
                            },
                            labelStyle={"display": "block", "marginBottom": "10px"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#f8f9fa",
                        "padding": "20px",
                        "borderRadius": "5px",
                        "border": "1px solid #dee2e6",
                        "marginBottom": "20px",
                    },
                )

                # Create initial plots
                dcc.Graph(
                    id="accuracy-plot",
                    figure=self.create_metric_comparison_plot(task, selected_dataset, "accuracy"),
                )

                dcc.Graph(
                    id="latency-plot",
                    figure=self.create_metric_comparison_plot(task, selected_dataset, "latency"),
                )

                dcc.Graph(
                    id="scatter-plot",
                    figure=self.create_latency_accuracy_scatter(task, selected_dataset),
                )

                return html.Div(
                    [
                        metric_buttons,
                        html.Div(id="dynamic-plot"),
                    ],
                )
            except ValueError:
                return html.Div(f"Invalid task: {selected_task}")

        @self.app.callback(
            Output("dynamic-plot", "children"),
            [Input("metric-selector", "value"), Input("dataset-tabs", "value"), Input("task-tabs", "value")],
        )
        def update_dynamic_plot(metric_type: str, selected_dataset: str, selected_task: str) -> html.Div:  # noqa: PLR0911
            if not selected_dataset or not selected_task or not metric_type:
                return html.Div()

            try:
                task = OTXTaskType(selected_task)

                # Check if "Average" tab is selected
                if selected_dataset == "average":
                    return self.create_averaged_plots(task, metric_type)

                # Regular dataset-specific plots
                if metric_type == "accuracy":
                    # Use the new method that creates multiple accuracy plots
                    return self.create_accuracy_comparison_plots(task, selected_dataset)
                if metric_type == "latency":
                    # Use the new method that creates multiple latency plots
                    return self.create_latency_comparison_plots(task, selected_dataset)
                if metric_type == "training_time":
                    return self.create_training_time_comparison_plot(task, selected_dataset)
                if metric_type == "epoch":
                    return self.create_epoch_comparison_plot(task, selected_dataset)
                if metric_type == "iter_time":
                    return self.create_iter_time_comparison_plot(task, selected_dataset)
                if metric_type == "gpu_mem":
                    return self.create_gpu_memory_comparison_plot(task, selected_dataset)
                if metric_type == "scatter":
                    fig = self.create_latency_accuracy_scatter(task, selected_dataset)
                    return dcc.Graph(figure=fig)
                return html.Div("Invalid metric type selected")

            except ValueError:
                return html.Div(f"Error creating plot for {selected_task}")

    def run(self, debug: bool = True, port: int = 8050, host: str = "127.0.0.1") -> None:
        """Run the Dash app."""
        logger.info("Loading initial benchmark data...")
        self.load_all_benchmark_data()
        logger.info(f"Starting Dash app on port {port}")
        self.app.run(debug=debug, port=port, host=host)


def main():
    parser = argparse.ArgumentParser(description="OTX Benchmark Dashboard")
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        help="Path to the benchmark directory",
        default=Path("tests/perf_v2/perf_history"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the dashboard on",
    )

    args = parser.parse_args()

    if not args.benchmark_dir.exists():
        logger.error(f"Benchmark directory does not exist: {args.benchmark_dir}")
        return

    dashboard = BenchmarkDashboard(args.benchmark_dir)
    # Bind to localhost only to avoid S104 Possible binding to all interfaces
    dashboard.run(debug=True, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
