# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DataValue(BaseModel):
    """A single data value (e.g., training dataset split)."""

    header: str = Field(..., description="Display name of the metric")
    key: str = Field(..., description="Metric key")
    value: float = Field(..., description="Metric value")


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


class TextMetric(BaseModel):
    """A text-based metric (e.g., training duration)."""

    header: str = Field(..., description="Display name of the metric")
    type: Literal["text"] = Field(default="text", description="Metric type")
    key: str = Field(..., description="Metric key")
    value: str = Field(..., description="Metric value as text")


class BarMetric(BaseModel):
    """A bar-based metric (e.g., dataset split)."""

    header: str = Field(..., description="Display name of the metric")
    type: Literal["bar"] = Field(default="bar", description="Metric type")
    key: str = Field(..., description="Metric key")
    value: list[DataValue] = Field(..., description="Metric values")


class RadialBarMetric(BaseModel):
    """A radial bar-based metric (e.g., dataset split)."""

    header: str = Field(..., description="Display name of the metric")
    type: Literal["radial_bar"] = Field(default="radial_bar", description="Metric type")
    key: str = Field(..., description="Metric key")
    value: list[DataValue] = Field(..., description="Metric value")


class LineMetric(BaseModel):
    """A series metric (e.g., loss over time)."""

    header: str = Field(..., description="Display name of the metric")
    type: Literal["line"] = Field(default="line", description="Metric type")
    key: str = Field(..., description="Metric key")
    value: SeriesData = Field(..., description="List of data points")


TrainingMetrics = Annotated[BarMetric | TextMetric | RadialBarMetric | LineMetric, Field(discriminator="type")]


class TrainingMetricsView(BaseModel):
    """Response model for Training Metrics endpoint."""

    training_metrics: list[TrainingMetrics] = Field(..., description="List of model training metrics")

    model_config = {
        "json_schema_extra": {
            "example": {
                "training_metrics": [
                    {
                        "header": "Training date",
                        "type": "text",
                        "key": "Training date",
                        "value": "2026-01-14T11:27:02.736000+00:00",
                    },
                    {"header": "Training duration", "type": "text", "key": "Training duration", "value": "0:06:05"},
                    {
                        "header": "Dataset split",
                        "type": "bar",
                        "key": "Dataset split",
                        "value": [
                            {"header": "Training", "key": "Training", "value": 109, "color": None},
                            {"header": "Validation", "key": "Validation", "value": 31, "color": None},
                            {"header": "Test", "key": "Test", "value": 16, "color": None},
                        ],
                    },
                    {
                        "header": "F-measure",
                        "type": "bar",
                        "key": "F-measure",
                        "value": [
                            {"header": "validation", "key": "validation", "value": 0.89, "color": None},
                            {"header": "test", "key": "test", "value": 0.88, "color": None},
                        ],
                    },
                    {
                        "header": "F-measure per label (test)",
                        "type": "radial_bar",
                        "key": "F-measure per label (test)",
                        "value": [
                            {"header": "pedestrian", "key": "pedestrian", "value": 0.86, "color": "#f7dab3ff"},
                            {"header": "vehicle", "key": "vehicle", "value": 0.89, "color": "#25a18eff"},
                        ],
                    },
                    {
                        "header": "lr-SGD",
                        "type": "line",
                        "key": "lr-SGD",
                        "value": {
                            "x_axis_label": "Timestamp",
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
                                    "color": None,
                                }
                            ],
                        },
                    },
                ]
            }
        }
    }
