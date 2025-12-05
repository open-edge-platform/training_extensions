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
    """
    Represents a data processing pipeline within a project.

    A pipeline connects a data source, a model revision, and a sink to process and output data.
    It manages the flow of data through the system and tracks the pipeline's operational state.

    Attributes:
        project_id: UUID of the project this pipeline belongs to.
        source: The data source configuration, None if disconnected.
        sink: The data sink configuration, None if disconnected.
        model_revision: The model revision to use for processing, None if no model is selected.
        source_id: UUID reference to the source entity.
        sink_id: UUID reference to the sink entity.
        model_revision_id: UUID reference to the model revision entity.
        status: Current operational status of the pipeline (IDLE or RUNNING).
        data_collection_policies: List of policies governing data collection behavior during pipeline execution.

    Raises:
        ValueError: If attempting to set status to RUNNING when source, sink, or model is not configured.
    """

    project_id: UUID
    source: Source | None = None
    sink: Sink | None = None
    model_revision: ModelRevision | None = None
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
