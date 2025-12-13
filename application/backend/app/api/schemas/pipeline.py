# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field

from app.models import DataCollectionPolicy, ModelRevision, PipelineStatus

from .sink import SinkView
from .source import SourceView


class PipelineView(BaseModel):
    project_id: UUID
    source: SourceView | None = None  # None if disconnected
    sink: SinkView | None = None  # None if disconnected
    model_revision: ModelRevision | None = Field(default=None, serialization_alias="model")
    status: PipelineStatus = PipelineStatus.IDLE
    data_collection_policies: list[DataCollectionPolicy] = Field(default_factory=list)

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
                    },
                    {
                        "type": "confidence_threshold",
                        "enabled": "true",
                        "confidence_threshold": 0.2,
                        "min_sampling_interval": 2.5,
                    },
                ],
            }
        }
    }
