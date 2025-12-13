# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.models import Pipeline, PipelineStatus
from app.repositories import PipelineRepository
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus, EventType
from app.services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(self, event_bus: EventBus, db_session: Session) -> None:
        self._event_bus: EventBus = event_bus
        self._db_session: Session = db_session

    def create_pipeline(self, project_id: UUID) -> Pipeline:
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline_db = PipelineDB(
            project_id=str(project_id),
        )
        created = pipeline_repo.save(pipeline_db)
        return Pipeline.model_validate(created)

    def get_active_pipeline(self) -> Pipeline | None:
        """Retrieve an active pipeline."""
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline = pipeline_repo.get_active_pipeline()
        return Pipeline.model_validate(pipeline) if pipeline is not None else None

    def get_pipeline_by_id(self, project_id: UUID) -> Pipeline:
        """Retrieve a pipeline by project ID."""
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline_db = pipeline_repo.get_by_id(str(project_id))
        if not pipeline_db:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(project_id))
        return Pipeline.model_validate(pipeline_db)

    def is_running(self, project_id: UUID) -> bool:
        """Retrieve a pipeline status by project ID."""
        pipeline_repo = PipelineRepository(self._db_session)
        return pipeline_repo.is_running(str(project_id))

    @parent_process_only
    def update_pipeline(self, project_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        pipeline = self.get_pipeline_by_id(project_id)
        to_update = type(pipeline).model_validate(pipeline.model_copy(update=partial_config))
        pipeline_repo = PipelineRepository(self._db_session)
        to_update_db = PipelineDB(
            project_id=str(to_update.project_id),
            source_id=str(to_update.source_id) if to_update.source_id else None,
            sink_id=str(to_update.sink_id) if to_update.sink_id else None,
            model_revision_id=str(to_update.model_revision_id) if to_update.model_revision_id else None,
            is_running=to_update.status.as_bool,
            data_collection_policies=[obj.model_dump() for obj in to_update.data_collection_policies],
        )
        pipeline_db = pipeline_repo.update(to_update_db)
        updated = Pipeline.model_validate(pipeline_db)
        if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
            # If the pipeline source_id or sink_id is being updated while running
            if pipeline.source.id != updated.source.id:  # type: ignore[union-attr] # source is always there for running pipeline
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            if pipeline.sink_id != updated.sink_id:  # type: ignore[union-attr] # sink is always there for running pipeline
                self._event_bus.emit_event(EventType.SINK_CHANGED)
            if pipeline.data_collection_policies != updated.data_collection_policies:
                self._event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)
        elif pipeline.status != updated.status:
            # If the pipeline is being activated or stopped
            self._event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)
        return updated
