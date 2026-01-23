# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, model_validator

from app.models import EvaluationResult


class MetricView(BaseModel):
    """
    Represents a single evaluation metric with its name and value.

    Attributes:
        name: The metric name (e.g., "accuracy", "f1_score", "precision").
        value: The metric value as a floating-point number.
    """

    name: str
    value: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "accuracy",
                "value": 0.97,
            }
        }
    }


class EvaluationView(BaseModel):
    """
    Represents evaluation results for a dataset revision subset.

    This view provides a summary of model evaluation metrics obtained for a specific
    dataset revision and subset combination.

    Attributes:
        dataset_revision_id: Unique identifier of the evaluated dataset revision.
        subset: The dataset subset used for evaluation (e.g., "training", "validation", "testing").
        metrics: List of evaluation metrics with their corresponding values.
    """

    dataset_revision_id: UUID
    subset: str
    metrics: list[MetricView]

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_revision_id": "3c6c6d38-1cd8-4458-b759-b9880c048b78",
                "subset": "testing",
                "metrics": [
                    {
                        "name": "accuracy",
                        "value": 0.97,
                    },
                    {
                        "name": "precision",
                        "value": 0.98,
                    },
                    {
                        "name": "recall",
                        "value": 0.94,
                    },
                ],
            }
        }
    }

    @model_validator(mode="before")
    @classmethod
    def populate_metrics(cls, data: object) -> object:
        if isinstance(data, EvaluationResult):
            return {
                "dataset_revision_id": data.dataset_revision_id,
                "subset": data.subset.value,
                "metrics": [MetricView(name=m[0], value=m[1]) for m in data.metrics.items()],
            }
        if isinstance(data, dict):
            return {
                "dataset_revision_id": data["dataset_revision_id"],
                "subset": data["subset"].value,
                "metrics": [MetricView(name=m[0], value=m[1]) for m in data["metrics"].items()],
            }
        return data
