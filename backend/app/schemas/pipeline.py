from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import model_validator

from app.schemas.base import BaseIDNameModel


class PipelineStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"

    @classmethod
    def from_bool(cls, is_running: bool) -> "PipelineStatus":
        return cls.RUNNING if is_running else cls.IDLE

    @property
    def as_bool(self) -> bool:
        return self == PipelineStatus.RUNNING


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

    @model_validator(mode="before")
    def set_status_from_is_running(cls, data: Any) -> Any:
        if hasattr(data, "is_running") and not hasattr(data, "status"):
            status = PipelineStatus.from_bool(getattr(data, "is_running"))
            d = data.__dict__.copy()
            d["status"] = status
            return d
        return data

    @model_validator(mode="after")
    def validate_running_status(self) -> "Pipeline":
        if self.status == PipelineStatus.RUNNING and any(
            x is None for x in (self.source_id, self.sink_id, self.model_id)
        ):
            raise ValueError(
                "Pipeline cannot be in 'running' status when source_id, sink_id, or model_id is not configured."
            )
        return self
