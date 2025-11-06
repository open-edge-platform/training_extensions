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
    """Format of the image related to the dataset item."""

    JPG = "jpg"
    PNG = "png"


class DatasetItemSubset(StrEnum):
    """Subset of the dataset item."""

    UNASSIGNED = "unassigned"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class DatasetItemAnnotation(BaseModel):
    """
    DatasetItemAnnotation represents an individual annotation within a dataset item.

    An annotation consists of a shape, one or more labels associated with that shape,
    and optionally a confidence score for each label (if applicable, e.g., for model predictions).

    Attributes:
        labels: A list of references to labels associated with the annotation.
        shape: The geometric shape defining the annotation area.
        confidences: A list of confidence scores corresponding to each label (if applicable).
    """

    shape: Shape
    labels: list[LabelReference]
    confidences: list[float] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
            }
        }
    }


class DatasetItem(BaseEntity):
    """
    DatasetItem represents an individual item within a dataset.

    Attributes:
        id: Unique identifier for the dataset item.
        project_id: Identifier of the project to which the dataset item belongs.
        name: Name of the dataset item.
        format: Format of the dataset item (e.g., JPG, PNG).
        width: Width of the dataset item in pixels.
        height: Height of the dataset item in pixels.
        size: Size of the dataset item in bytes.
        annotation_data: List of annotations associated with the dataset item.
        user_reviewed: Indicates whether the dataset item has been reviewed by a user,
            namely if its annotation has been created/accepted by a human or if it is just a raw model prediction.
        prediction_model_id: Identifier of the model that generated predictions for this dataset item, if applicable.
        source_id: Identifier of the source from which the dataset item was acquired, if applicable.
        subset: Subset to which the dataset item belongs (e.g., training, validation, testing).
        subset_assigned_at: Timestamp indicating when the dataset item was assigned to its subset.
    """

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
