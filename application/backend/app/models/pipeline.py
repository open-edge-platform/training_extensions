# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from .base import BaseEntity
from .data_collection_policy import DataCollectionPolicy
from .model_revision import ModelRevision
from .sink import Sink
from .source import Source


class PipelineStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"

    @classmethod
    def from_bool(cls, is_running: bool) -> "PipelineStatus":
        return cls.RUNNING if is_running else cls.IDLE

    @property
    def as_bool(self) -> bool:
        return self == PipelineStatus.RUNNING


class Pipeline(BaseEntity):
    project_id: UUID  # ID of the project this pipeline belongs to
    source: Source | None = None  # None if disconnected
    sink: Sink | None = None  # None if disconnected
    model_revision: ModelRevision | None = None  # None if no model is selected
    source_id: UUID | None = None
    sink_id: UUID | None = None
    model_revision_id: UUID | None = None
    status: PipelineStatus = PipelineStatus.IDLE
    data_collection_policies: list[DataCollectionPolicy] = Field(default_factory=list)

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
            x is None for x in (self.source_id, self.sink_id, self.model_revision_id)
        ):
            raise ValueError("Pipeline cannot be in 'running' state when source, sink, or model is not configured.")
        return self
