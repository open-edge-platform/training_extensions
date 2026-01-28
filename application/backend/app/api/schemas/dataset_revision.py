# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field

from app.core.models import BaseIDModel


class ItemCount(BaseModel):
    total: int = Field(..., description="Total number of items in the dataset")
    training: int = Field(..., description="Number of items in the training subset")
    validation: int = Field(..., description="Number of items in the validation subset")
    testing: int = Field(..., description="Number of items in the testing subset")


class DatasetRevisionView(BaseIDModel):
    """Represents a dataset revision, including its project association, display name, file deletion status, and item counts."""

    project_id: UUID = Field(..., description="Id of the project of the dataset revision")
    name: str = Field(..., description="User friendly model name")
    files_deleted: bool = Field(..., description="Whether or not files are deleted from disk for this dataset revision")
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
