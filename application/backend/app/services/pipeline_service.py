# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.repositories import PipelineRepository
from app.schemas import PipelineStatus, PipelineView
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus, EventType
from app.services.mappers import PipelineMapper
from app.services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(self, event_bus: EventBus, db_session: Session) -> None:
        self._event_bus: EventBus = event_bus
        self._db_session: Session = db_session

    def create_pipeline(self, project_id: UUID) -> PipelineView:
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline_db = PipelineDB(
            project_id=str(project_id),
        )
        return PipelineMapper.to_schema(pipeline_repo.save(pipeline_db))

    def get_active_pipeline(self) -> PipelineView | None:
        """Retrieve an active pipeline."""
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline = pipeline_repo.get_active_pipeline()
        return PipelineMapper.to_schema(pipeline) if pipeline is not None else None

    def get_pipeline_by_id(self, project_id: UUID) -> PipelineView:
        """Retrieve a pipeline by project ID."""
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline = pipeline_repo.get_by_id(str(project_id))
        if not pipeline:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(project_id))
        return PipelineMapper.to_schema(pipeline)

    def is_running(self, project_id: UUID) -> bool:
        """Retrieve a pipeline status by project ID."""
        pipeline_repo = PipelineRepository(self._db_session)
        return pipeline_repo.is_running(str(project_id))

    @parent_process_only
    def update_pipeline(self, project_id: UUID, partial_config: dict) -> PipelineView:
        """Update an existing pipeline."""
        pipeline = self.get_pipeline_by_id(project_id)
        to_update = type(pipeline).model_validate(pipeline.model_copy(update=partial_config))
        pipeline_repo = PipelineRepository(self._db_session)
        updated = PipelineMapper.to_schema(pipeline_repo.update(PipelineMapper.from_schema(to_update)))
        if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
            # If the pipeline source_id or sink_id is being updated while running
            if pipeline.source.id != updated.source.id:  # type: ignore[union-attr] # source is always there for running pipeline
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            if pipeline.sink.id != updated.sink.id:  # type: ignore[union-attr] # sink is always there for running pipeline
                self._event_bus.emit_event(EventType.SINK_CHANGED)
            if pipeline.data_collection_policies != updated.data_collection_policies:
                self._event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)
        elif pipeline.status != updated.status:
            # If the pipeline is being activated or stopped
            self._event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)
        return updated
