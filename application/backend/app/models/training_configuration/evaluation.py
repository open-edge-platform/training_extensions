#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field


class ValidationMetric(StrEnum):
    """Available metrics for model evaluation during validation."""

    DEFAULT = "default"  # Auto-select the most appropriate metric based on the task type
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F_MEASURE = "F-measure"
    MAP = "mAP"  # mAP@0.5:0.95
    MAP_50 = "mAP@0.5"
    MAP_75 = "mAP@0.75"
    MAR_1 = "mAR@1"
    MAR_10 = "mAR@10"
    MAR_100 = "mAR@100"


class TaskLevelEvaluationParameters(BaseModel):
    """Evaluation parameters that apply at the task level and are relevant for all models architectures."""

    validation_metric: ValidationMetric = Field(
        default=ValidationMetric.DEFAULT,
        title="Validation metric",
        description="Metric used to evaluate model performance during validation.",
    )
