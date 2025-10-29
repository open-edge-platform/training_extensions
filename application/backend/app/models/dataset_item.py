# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from .base import BaseEntity
from .label import LabelReference
from .shape import Shape


class DatasetItemFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"


class DatasetItemSubset(StrEnum):
    UNASSIGNED = "unassigned"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class DatasetItemAnnotation(BaseModel):
    """
    Dataset item annotation
    """

    labels: list[LabelReference]
    shape: Shape
    confidence: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
            }
        }
    }


class DatasetItem(BaseEntity):
    id: UUID
    project_id: UUID
    name: str
    format: DatasetItemFormat
    width: int
    height: int
    size: int
    annotation_data: list[DatasetItemAnnotation] | None
    user_reviewed: bool
    prediction_model_id: UUID | None
    source_id: UUID | None
    subset: DatasetItemSubset | None
    subset_assigned_at: datetime | None
