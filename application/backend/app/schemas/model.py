# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.base import BaseIDModel


class ModelFormat(StrEnum):
    OPENVINO = "openvino_ir"
    ONNX = "onnx"


class TrainingStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    SUCCESSFUL = "successful"


class TrainingInfo(BaseModel):
    """Information about the training process of a model revision."""

    status: TrainingStatus = Field(description="Training status", default=TrainingStatus.NOT_STARTED)
    label_schema_revision: dict = Field(description="Label schema revision used for training")
    configuration: dict = Field(description="Training configuration parameters")
    start_time: datetime | None = Field(None, description="Training start time")
    end_time: datetime | None = Field(None, description="Training end time")
    dataset_revision_id: UUID | None = Field(None, description="Dataset revision ID used for training")


class Model(BaseIDModel):
    """Represents a model revision with its architecture, parent revision, training info, and file status."""

    architecture: str = Field(..., description="Model architecture name")
    parent_revision: UUID | None = Field(None, description="Parent model revision ID")
    training_info: TrainingInfo = Field(..., description="Information about the training process")
    files_deleted: bool = Field(description="Indicates if model files have been deleted", default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "architecture": "Object_Detection_YOLOX_X",
                "parent_revision": "06091f82-5506-41b9-b97f-c761380df870",
                "training_info": {
                    "status": "in_progress",
                    "start_time": "2021-06-29T16:24:30.928000+00:00",
                    "end_time": "2021-06-29T16:24:30.928000+00:00",
                    "dataset_revision_id": "3c6c6d38-1cd8-4458-b759-b9880c048b78",
                    "label_schema_revision": {},
                    "configuration": {},
                },
                "files_deleted": False,
            }
        }
    }
