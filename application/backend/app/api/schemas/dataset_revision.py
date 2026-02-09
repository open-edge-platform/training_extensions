# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import BaseModel, Field

from app.core.models import BaseIDModel


class ItemCount(BaseModel):
    """Counts of samples in different splits of a dataset revision."""

    total: int = Field(..., description="Total number of items in the dataset")
    training: int = Field(..., description="Number of items in the training subset")
    validation: int = Field(..., description="Number of items in the validation subset")
    testing: int = Field(..., description="Number of items in the testing subset")


class DatasetRevisionView(BaseIDModel):
    """A dataset revision is an immutable snapshot of a training dataset."""

    name: str = Field(..., description="Name of the dataset revision")
    created_at: datetime = Field(..., description="Timestamp when the dataset revision was created")
    files_deleted: bool = Field(..., description="Indicates if the dataset revision files have been deleted")
    item_counts: ItemCount = Field(..., description="Number of items in the dataset (null for deleted datasets)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "1ed5487a-a9e7-44d9-aaba-ea92fb8983ee",
                "name": "Dataset (1ed5487a)",
                "created_at": "2023-10-01T12:00:00Z",
                "files_deleted": False,
                "item_counts": {"total": 100, "training": 70, "validation": 20, "testing": 10},
            }
        }
    }
