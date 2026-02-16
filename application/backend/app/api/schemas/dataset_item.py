# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.core.models import BaseRequiredIDModel, Pagination
from app.models import DatasetItemAnnotation, DatasetItemSubset, MediaFormat


class DatasetRevisionItemView(BaseRequiredIDModel):
    """
    Dataset Revision item
    """

    format: MediaFormat
    width: int
    height: int
    subset: DatasetItemSubset

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "format": "jpg",
                "width": 1280,
                "height": 720,
                "subset": "unassigned",
            }
        }
    }


class DatasetItemView(BaseRequiredIDModel):
    """
    Dataset item
    """

    subset: DatasetItemSubset
    user_reviewed: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "user_reviewed": False,
                "subset": "training",
            }
        }
    }


class SetDatasetItemAnnotations(BaseModel):
    """Schema for setting dataset item annotations"""

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


class DatasetItemAnnotations(BaseModel):
    """
    Dataset item annotations with information about source
    """

    annotations: list[DatasetItemAnnotation]
    user_reviewed: bool
    prediction_model_id: UUID | None = None

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

    items: list[DatasetItemView]
    pagination: Pagination


class DatasetRevisionItemsWithPagination(BaseModel):
    """
    Dataset Revision Items list with pagination info
    """

    items: list[DatasetRevisionItemView]
    pagination: Pagination


class DatasetItemAssignSubset(BaseModel):
    """Schema for assigning a subset to dataset item"""

    subset: Literal[DatasetItemSubset.TESTING, DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION]

    model_config = {"json_schema_extra": {"example": {"subset": "training"}}}
