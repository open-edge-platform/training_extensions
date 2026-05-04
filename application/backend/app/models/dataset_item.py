# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from .base import BaseEntity
from .label import LabelReference
from .shape import Shape


class DatasetItemSubset(StrEnum):
    """Subset of the dataset item."""

    UNASSIGNED = "unassigned"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class DatasetItemAnnotationStatus(StrEnum):
    """
    Annotation status filter for dataset items.

    Used to filter media based on their annotation state:

    - ``WITH_ANNOTATIONS``: Matches images that have annotations, or videos that have at least one annotated frame.
    - ``MISSING_ANNOTATIONS``: Matches images that have no annotations, or videos that have at least one unannotated
        frame.
    """

    WITH_ANNOTATIONS = "with_annotations"
    MISSING_ANNOTATIONS = "missing_annotations"


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
        id: Unique identifier for the dataset item, is equal to the corresponding media.
        project_id: Identifier of the project to which the dataset item belongs.
        annotation_data: List of annotations associated with the dataset item.
        user_reviewed: Flag indicating whether the dataset item requires the user's attention before it can be used
            for training. Typically, it is false for unannotated media (since they need to be annotated) and model
            predictions (since they need to be reviewed). It is true for items with user-submitted annotations, or
            model predictions that have been accepted (with or without modification) by a user.
        prediction_model_id: Identifier of the model that generated predictions for this dataset item, if applicable.
        subset: Subset to which the dataset item belongs (e.g., training, validation, testing).
        subset_assigned_at: Timestamp indicating when the dataset item was assigned to its subset.
    """

    id: UUID
    project_id: UUID
    annotation_data: list[DatasetItemAnnotation] | None
    user_reviewed: bool
    prediction_model_id: UUID | None
    subset: DatasetItemSubset = DatasetItemSubset.UNASSIGNED
    subset_assigned_at: datetime | None
