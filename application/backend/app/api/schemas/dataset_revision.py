# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.core.models import BaseIDModel


class ItemCount(BaseIDModel):
    total: int = Field(..., description="Total number of items in the dataset")
    training: int = Field(..., description="Number of items in the training subset")
    validation: int = Field(..., description="Number of items in the validation subset")
    testing: int = Field(..., description="Number of items in the testing subset")


class DatasetRevisionView(BaseIDModel):
    """Represents a model revision with its architecture, parent revision, training info, variants, and file status."""

    project_id: UUID = Field(..., description="Id of the project of the dataset revision")
    name: str = Field(..., description="User friendly model name")
    created_at: datetime = Field(..., description="Creation date of the dataset revision")
    item_counts: ItemCount = Field(description="Number of items in the dataset")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "1ed5487a-a9e7-44d9-aaba-ea92fb8983ee",
                "project_id": "f619c2c8-b14c-4188-9268-d2e0d0662bdd",
                "name": "Dataset (1ed5487a)",
                "created_at": "2023-10-01T12:00:00Z",
                "item_counts": {"total": 100, "training": 70, "validation": 20, "testing": 10},
            }
        }
    }
