# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """A single data point in a series metric."""

    x: float = Field(..., description="X-axis value (e.g., epoch number)")
    y: float = Field(..., description="Y-axis value (e.g., loss value)")
    type: Literal["point"] = Field(..., description="Metric type")


class LineData(BaseModel):
    """A series line data (e.g., dataset split)."""

    header: str = Field(..., description="Display name of the metric")
    key: str = Field(..., description="Metric key")
    points: list[DataPoint] = Field(..., description="List of data points")


class SeriesData(BaseModel):
    """A series metric (e.g., loss over time)."""

    x_axis_label: str = Field(..., description="Label for the X-axis")
    y_axis_label: str = Field(..., description="Label for the Y-axis")
    line_data: list[LineData] = Field(..., description="List of data points")


class LineMetric(BaseModel):
    """A series metric (e.g., loss over time)."""

    header: str = Field(..., description="Display name of the metric")
    type: Literal["line"] = Field(default="line", description="Metric type")
    key: str = Field(..., description="Metric key")
    value: SeriesData = Field(..., description="List of data points")


class TrainingMetricsView(BaseModel):
    """Response model for Training Metrics endpoint."""

    training_metrics: list[LineMetric] = Field(..., description="List of model training metrics")

    model_config = {
        "json_schema_extra": {
            "example": {
                "training_metrics": [
                    {
                        "header": "lr-SGD",
                        "type": "line",
                        "key": "lr-SGD",
                        "value": {
                            "x_axis_label": "Step",
                            "y_axis_label": "lr-SGD",
                            "line_data": [
                                {
                                    "header": "lr-SGD",
                                    "key": "lr-SGD",
                                    "points": [
                                        {"x": 1, "y": 0.001, "type": "point"},
                                        {"x": 2, "y": 0.001, "type": "point"},
                                        {"x": 3, "y": 0.001, "type": "point"},
                                    ],
                                }
                            ],
                        },
                    },
                ]
            }
        }
    }
