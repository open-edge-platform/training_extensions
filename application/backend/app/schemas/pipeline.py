# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.schemas.model import Model
from app.schemas.sink import Sink
from app.schemas.source import Source


class PipelineStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"

    @classmethod
    def from_bool(cls, is_running: bool) -> "PipelineStatus":
        return cls.RUNNING if is_running else cls.IDLE

    @property
    def as_bool(self) -> bool:
        return self == PipelineStatus.RUNNING


class DataCollectionPolicyBase(BaseModel):
    type: str
    enabled: bool = True


class FixedRateDataCollectionPolicy(DataCollectionPolicyBase):
    type: Literal["fixed_rate"] = "fixed_rate"
    rate: float


DataCollectionPolicy = Annotated[FixedRateDataCollectionPolicy, Field(discriminator="type")]


class PipelineView(BaseModel):
    project_id: UUID  # ID of the project this pipeline belongs to
    source: Source | None = None  # None if disconnected
    sink: Sink | None = None  # None if disconnected
    model: Model | None = None  # None if no model is selected
    source_id: UUID | None = Field(
        default=None, exclude=True
    )  # ID of the source, used for DB mapping, not exposed in API
    sink_id: UUID | None = Field(default=None, exclude=True)  # ID of the sink, used for DB mapping, not exposed in API
    model_id: UUID | None = Field(
        default=None, exclude=True
    )  # ID of the model, used for DB mapping, not exposed in API
    status: PipelineStatus = PipelineStatus.IDLE  # Current status of the pipeline
    data_collection_policies: list[DataCollectionPolicy] = []  # List of data collection policies

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "source": {
                    "source_type": "video_file",
                    "name": "Sample Video",
                    "id": "712750b2-5a82-47ee-8fba-f3dc96cb615d",
                    "video_path": "/path/to/video.mp4",
                },
                "sink": {
                    "id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
                    "sink_type": "folder",
                    "name": "Local Folder",
                    "folder_path": "/path/to/output",
                    "output_formats": ["image_original", "image_with_predictions", "predictions"],
                    "rate_limit": 0.2,
                },
                "model": {
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
                },
                "status": "running",
                "data_collection_policies": [
                    {
                        "type": "fixed_rate",
                        "enabled": "true",
                        "rate": 0.02,
                    }
                ],
            }
        }
    }

    @model_validator(mode="before")
    def set_status_from_is_running(cls, data: Any) -> Any:
        if hasattr(data, "is_running") and not hasattr(data, "status"):
            status = PipelineStatus.from_bool(getattr(data, "is_running"))
            d = data.__dict__.copy()
            d["status"] = status
            return d
        return data

    @model_validator(mode="after")
    def validate_running_status(self) -> "PipelineView":
        if self.status == PipelineStatus.RUNNING and any(
            x is None for x in (self.source_id, self.sink_id, self.model_id)
        ):
            raise ValueError("Pipeline cannot be in 'running' status when source, sink, or model is not configured.")
        return self
