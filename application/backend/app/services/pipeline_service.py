# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.models import Pipeline, PipelineStatus
from app.repositories import PipelineRepository
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus, EventType
from app.services.parent_process_guard import parent_process_only

from . import BaseSessionManagedService
from .system_service import DEFAULT_DEVICE, SystemService

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class OtherProjectActiveError(Exception):
    """
    Exception raised when trying to run a pipeline in one project, while a pipeline of another project is still running.
    """

    def __init__(self, requested_project_id: str, active_project_id: str):
        super().__init__(
            f"Attempted to enable a pipeline in project with ID {requested_project_id}, while a pipeline is still "
            f"enabled in another project with ID {active_project_id}. Please first disable pipeline in project with "
            f"ID {active_project_id}"
        )


class PipelineService(BaseSessionManagedService):
    def __init__(
        self, system_service: SystemService, event_bus: EventBus | None = None, db_session: Session | None = None
    ) -> None:
        super().__init__(db_session)
        self._event_bus: EventBus | None = event_bus
        self._system_service: SystemService = system_service

    def create_pipeline(self, project_id: UUID) -> Pipeline:
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = PipelineDB(
            project_id=str(project_id),
        )
        created = pipeline_repo.save(pipeline_db)
        return Pipeline.model_validate(created)

    def get_active_pipeline(self) -> Pipeline | None:
        """Retrieve an active pipeline."""
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = pipeline_repo.get_active_pipeline()
        if pipeline_db is None:
            return None

        if not self._system_service.validate_device(pipeline_db.device):
            logger.warning(
                "The configured device '{}' is not available for pipeline '{}'. Falling back to 'cpu'.",
                pipeline_db.device,
                pipeline_db.project_id,
            )
            pipeline_db.device = DEFAULT_DEVICE
            pipeline_repo.update(pipeline_db)
        return Pipeline.model_validate(pipeline_db)

    def get_pipeline_by_id(self, project_id: UUID) -> Pipeline:
        """Retrieve a pipeline by project ID."""
        pipeline_repo = PipelineRepository(self.db_session)
        pipeline_db = pipeline_repo.get_by_id(str(project_id))
        if not pipeline_db:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(project_id))
        return Pipeline.model_validate(pipeline_db)

    def is_running(self, project_id: UUID) -> bool:
        """Retrieve a pipeline status by project ID."""
        pipeline_repo = PipelineRepository(self.db_session)
        return pipeline_repo.is_running(str(project_id))

    @parent_process_only
    def update_pipeline(self, project_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        pipeline = self.get_pipeline_by_id(project_id)
        base = pipeline.model_dump()
        to_update = type(pipeline).model_validate({**base, **partial_config})
        pipeline_repo = PipelineRepository(self.db_session)
        to_update_db = PipelineDB(
            project_id=str(to_update.project_id),
            source_id=str(to_update.source_id) if to_update.source_id else None,
            sink_id=str(to_update.sink_id) if to_update.sink_id else None,
            model_revision_id=str(to_update.model_id) if to_update.model_id else None,
            is_running=to_update.status.as_bool,
            data_collection=to_update.data_collection.model_dump(),
            device=to_update.device,
        )
        if to_update_db.is_running:
            # Only one pipeline can run at the same time. Note that only one pipeline per project exists.
            active_pipeline_db = pipeline_repo.get_active_pipeline()
            if active_pipeline_db is not None and to_update_db.project_id != active_pipeline_db.project_id:
                raise OtherProjectActiveError(
                    requested_project_id=to_update_db.project_id, active_project_id=active_pipeline_db.project_id
                )
        pipeline_db = pipeline_repo.update(to_update_db)
        updated = Pipeline.model_validate(pipeline_db)
        self.__emit_event(pipeline, updated)
        return updated

    def __emit_event(self, pipeline: Pipeline, updated: Pipeline) -> None:
        if self._event_bus is None:
            raise ValueError(
                "Event bus is required to update pipeline. This is because updating pipeline may trigger events that "
                "require other services to react."
            )
        if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
            # If the pipeline source_id or sink_id is being updated while running
            if pipeline.source.id != updated.source.id:  # type: ignore[union-attr] # source is always there for running pipeline
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            if pipeline.sink_id != updated.sink_id:  # type: ignore[union-attr] # sink is always there for running pipeline
                self._event_bus.emit_event(EventType.SINK_CHANGED)
            if pipeline.data_collection != updated.data_collection:
                self._event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)
            if pipeline.device != updated.device:
                self._event_bus.emit_event(EventType.INFERENCE_DEVICE_CHANGED)
            if pipeline.model_id != updated.model_revision.id:  # type: ignore[union-attr] # model_revision is always there for running pipeline
                self._event_bus.emit_event(EventType.MODEL_CHANGED)
        elif pipeline.status != updated.status:
            # If the pipeline is being activated or stopped
            self._event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)
