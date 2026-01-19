# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from app.db.schema import ModelRevisionDB
from app.models.base import BaseEntity


class ModelFormat(StrEnum):
    OPENVINO = "openvino"
    ONNX = "onnx"
    PYTORCH = "pytorch"


class ModelPrecision(StrEnum):
    FP16 = "fp16"
    FP32 = "fp32"


class TrainingStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    SUCCESSFUL = "successful"


class TrainingInfo(BaseEntity):
    """Information about the training process of a model revision."""

    status: TrainingStatus = TrainingStatus.NOT_STARTED
    label_schema_revision: dict = Field(default_factory=dict)
    configuration: dict = Field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    dataset_revision_id: UUID | None = None


class ModelRevision(BaseEntity):
    """
    Represents a specific revision of a machine learning model.

    A model revision tracks a particular version of a model, including its architecture, relationship to other
    revisions, training information, and file storage status.

    Attributes:
        id: Unique identifier for the model revision.
        name: User friendly name to identify a model
        architecture: Name of the model architecture (e.g., 'Object_Detection_RTDetr_50').
        parent_revision: UUID of the parent revision if this is derived from another revision,
            None if this is the initial revision.
        training_info: Details about the training process, including status, configuration, and associated dataset.
            None if training hasn't started.
        files_deleted: Flag indicating whether the model files have been deleted from storage.
    """

    id: UUID
    name: str
    architecture: str
    parent_revision: UUID | None = None
    training_info: TrainingInfo | None = None
    files_deleted: bool = False

    @model_validator(mode="before")
    @classmethod
    def populate_training_info(cls, data: object) -> object:
        if isinstance(data, ModelRevisionDB):
            return {
                "id": data.id,
                "name": data.name,
                "architecture": data.architecture,
                "parent_revision": data.parent_revision,
                "files_deleted": data.files_deleted,
                "training_info": TrainingInfo.model_validate(data),
            }
        return data
