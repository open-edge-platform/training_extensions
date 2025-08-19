from uuid import UUID

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.db.schema import PipelineDB
from app.repositories import PipelineRepository
from app.schemas import Pipeline, PipelineStatus
from app.services import ActivePipelineService
from app.services.base import (
    GenericPersistenceService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ServiceConfig,
)
from app.services.mappers import PipelineMapper
from app.services.parent_process_guard import parent_process_only
from app.utils import Singleton

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService(metaclass=Singleton):
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Pipeline, PipelineDB, PipelineRepository] = (
            GenericPersistenceService(ServiceConfig(PipelineRepository, PipelineMapper, ResourceType.PIPELINE))
        )

    @staticmethod
    def _notify_source_changed() -> None:
        from app.core import Scheduler  # cyclic-import

        config_changed_condition = Scheduler().mp_config_changed_condition
        with config_changed_condition:
            config_changed_condition.notify_all()

    @staticmethod
    def _notify_sink_changed() -> None:
        ActivePipelineService().reload()

    @staticmethod
    def _notify_pipeline_changed() -> None:
        PipelineService._notify_source_changed()
        PipelineService._notify_sink_changed()

    def get_pipeline_by_id(self, pipeline_id: UUID, db: Session | None = None) -> Pipeline:
        """Retrieve a pipeline by its ID."""
        pipeline = self._persistence.get_by_id(pipeline_id, db)
        if not pipeline:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(pipeline_id))
        return pipeline

    def list_pipelines(self) -> list[Pipeline]:
        """List all pipelines."""
        return self._persistence.list_all()

    @parent_process_only
    def create_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Create a new pipeline."""
        with get_db_session() as db:
            created = self._persistence.create(pipeline, db)
            db.commit()
            if created.status == PipelineStatus.RUNNING:
                self._notify_source_changed()
                self._notify_sink_changed()
            return created

    @parent_process_only
    def update_pipeline(self, pipeline_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        with get_db_session() as db:
            pipeline = self.get_pipeline_by_id(pipeline_id, db)
            updated = self._persistence.update(pipeline, partial_config, db)
            db.commit()
            if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
                # If the pipeline source_id or sink_id is being updated while running
                if pipeline.source_id != updated.source_id:
                    self._notify_source_changed()
                if pipeline.sink_id != updated.sink_id:
                    self._notify_sink_changed()
            elif pipeline.status != updated.status:
                # If the pipeline is being activated or stopped
                self._notify_pipeline_changed()
            return updated

    @parent_process_only
    def delete_pipeline_by_id(self, pipeline_id: UUID) -> None:
        """Delete a pipeline by its ID."""
        with get_db_session() as db:
            pipeline = self.get_pipeline_by_id(pipeline_id, db)
            if pipeline.status == PipelineStatus.RUNNING:
                raise ResourceInUseError(ResourceType.PIPELINE, str(pipeline_id), MSG_ERR_DELETE_RUNNING_PIPELINE)
            self._persistence.delete_by_id(pipeline_id, db)
            db.commit()
