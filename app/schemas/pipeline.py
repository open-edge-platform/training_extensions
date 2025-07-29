from enum import Enum
from uuid import UUID

from app.schemas.base import BaseIDNameModel


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"


class Pipeline(BaseIDNameModel):
    source_id: UUID | None = None  # ID of the source; None if disconnected
    sink_id: UUID | None = None  # ID of the sink; None if disconnected
    model_id: UUID | None = None  # ID of the active model; None if no model is selected
    status: PipelineStatus = PipelineStatus.IDLE  # Current status of the pipeline

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "Production Pipeline",
                "source_id": "d2cbd8d0-17b8-463e-85a2-4aaed031674d",
                "sink_id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
                "model_id": "b0feaabc-da2b-442e-9b3e-55c11c2c2ff2",
                "status": "running",
            }
        }
    }
