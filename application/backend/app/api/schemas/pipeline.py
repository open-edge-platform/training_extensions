# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models import (
    DataCollectionConfig,
    InferenceWorkerStatus,
    ModelRevision,
    PipelineStatus,
    SinkStatus,
    SourceStatus,
)
from app.models.model_revision import ModelVariant

from .sink import SinkView
from .source import SourceView


class Status(BaseModel):
    status: str
    message: str | None
    timestamp: datetime

    @staticmethod
    def unavailable() -> Status:
        return Status(status="unavailable", message=None, timestamp=datetime.now())


class PipelineComponentsHealth(BaseModel):
    source: Status
    sink: Status
    model: Status


class PipelineHealth(BaseModel):
    status: str
    components: PipelineComponentsHealth | None = None

    @staticmethod
    def idle() -> PipelineHealth:
        return PipelineHealth(status=PipelineStatus.IDLE)

    @staticmethod
    def running(
        source_status: SourceStatus | None,
        sink_status: SinkStatus | None,
        inference_status: InferenceWorkerStatus | None,
    ) -> PipelineHealth:
        return PipelineHealth(
            status=PipelineStatus.RUNNING,
            components=PipelineComponentsHealth(
                source=Status(
                    status=source_status.code, message=source_status.message, timestamp=source_status.timestamp
                )
                if source_status
                else Status.unavailable(),
                sink=Status(status=sink_status.code, message=sink_status.message, timestamp=sink_status.timestamp)
                if sink_status
                else Status.unavailable(),
                model=Status(
                    status=inference_status.code, message=inference_status.message, timestamp=inference_status.timestamp
                )
                if inference_status
                else Status.unavailable(),
            ),
        )


class PipelineView(BaseModel):
    project_id: UUID
    source: SourceView | None = None  # None if disconnected
    sink: SinkView | None = None  # None if disconnected
    model_revision: ModelRevision | None = Field(default=None, serialization_alias="model")
    model_variant: ModelVariant | None = Field(default=None)
    status: PipelineStatus = PipelineStatus.IDLE
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    device: str = Field(default="cpu", description="Inference device (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')")

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
                    "architecture": "object-detection-yolox-x",
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
                "device": "cpu",
                "data_collection": {
                    "max_dataset_size": 500,
                    "policies": [
                        {
                            "type": "fixed_rate",
                            "enabled": True,
                            "rate": 0.02,
                        },
                        {
                            "type": "confidence_threshold",
                            "enabled": True,
                            "confidence_threshold": 0.2,
                            "min_sampling_interval": 2.5,
                        },
                    ],
                },
            }
        }
    }
