# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel

from app.core.models import BaseRequiredIDModel, Pagination
from app.models import DatasetItemSubset, MediaFormat


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
    user_reviewed: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "user_reviewed": False,
                "subset": "training",
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
