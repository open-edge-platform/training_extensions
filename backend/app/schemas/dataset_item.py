# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from app.schemas.base import BaseIDNameModel, Pagination


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


class DatasetItemsWithPagination(BaseModel):
    """
    Dataset Items list with pagination info
    """

    items: list[DatasetItem]
    pagination: Pagination
