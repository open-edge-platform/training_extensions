# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from app.schemas.base import BaseIDNameModel, Pagination
from app.schemas.label import LabelReference
from app.schemas.shape import Shape


class DatasetItemFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"


class DatasetItemSubset(StrEnum):
    UNASSIGNED = "unassigned"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class DatasetItem(BaseIDNameModel):
    """
    Dataset item
    """

    format: DatasetItemFormat
    width: int
    height: int
    size: int
    source_id: UUID | None = None
    subset: DatasetItemSubset

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "img-010203",
                "format": "jpg",
                "width": 1280,
                "height": 720,
                "size": 2211840,
                "source_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
                "subset": "unassigned",
            }
        }
    }


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


class DatasetItemAnnotations(BaseModel):
    """
    Dataset item annotations
    """

    annotations: list[DatasetItemAnnotation]

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotations": [
                    {
                        "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                        "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                    }
                ]
            }
        }
    }


class DatasetItemAnnotationsWithSource(DatasetItemAnnotations):
    """
    Dataset item annotations with information about source
    """

    user_reviewed: bool
    prediction_model_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotations": [
                    {
                        "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                        "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                    }
                ],
                "user_reviewed": "false",  # type: ignore[dict-item]
                "prediction_model_id": "null",  # type: ignore[dict-item]
            }
        }
    }


class DatasetItemsWithPagination(BaseModel):
    """
    Dataset Items list with pagination info
    """

    items: list[DatasetItem]
    pagination: Pagination
