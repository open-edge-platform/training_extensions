# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import Field

from app.core.models import BaseIDModel
from app.models import TrainingInfo


class ModelView(BaseIDModel):
    """Represents a model revision with its architecture, parent revision, training info, and file status."""

    name: str = Field(..., description="User friendly model name")
    architecture: str = Field(..., description="Model architecture name")
    parent_revision: UUID | None = Field(None, description="Parent model revision ID")
    training_info: TrainingInfo = Field(..., description="Information about the training process")
    files_deleted: bool = Field(description="Indicates if model files have been deleted", default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "name": "Object_Detection_YOLOX_X (76e07d18)",
                "architecture": "Object_Detection_YOLOX_X",
                "parent_revision": "06091f82-5506-41b9-b97f-c761380df870",
                "training_info": {
                    "status": "in_progress",
                    "start_time": "2021-06-29T16:24:30.928000+00:00",
                    "end_time": "2021-06-29T16:24:30.928000+00:00",
                    "dataset_revision_id": "3c6c6d38-1cd8-4458-b759-b9880c048b78",
                    "label_schema_revision": {
                        "labels": [
                            {
                                "id": "a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29",
                                "name": "cat",
                            },
                            {
                                "id": "8aa85368-11ba-4507-88f2-6a6704d78ef5",
                                "name": "dog",
                            },
                        ]
                    },
                    "configuration": {},
                },
                "files_deleted": False,
            }
        }
    }
