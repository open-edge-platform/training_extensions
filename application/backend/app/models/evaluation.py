# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel

from app.models import DatasetItemSubset


class EvaluationResult(BaseModel):
    """
    Represents the evaluation results of a model revision on a specific dataset subset.

    This class encapsulates the performance metrics obtained when evaluating a model
    revision against a dataset revision's particular subset (e.g., train, validation, test).

    Attributes:
        model_revision_id: Unique identifier of the model revision being evaluated.
        dataset_revision_id: Unique identifier of the dataset revision used for evaluation.
        subset: The dataset subset used for evaluation (e.g., "training", "validation", "testing").
        metrics: Dictionary mapping metric names to their corresponding float scores
                 (e.g., {"accuracy": 0.95, "f1_score": 0.87}).
    """

    model_revision_id: UUID
    dataset_revision_id: UUID
    subset: DatasetItemSubset

    metrics: dict[str, float]
