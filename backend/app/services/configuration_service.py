import logging
from collections.abc import Callable
from enum import StrEnum
from uuid import UUID

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.db.schema import SinkDB, SourceDB
from app.repositories import PipelineRepository, SinkRepository, SourceRepository
from app.schemas import Sink, Source
from app.services import ActivePipelineService
from app.services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from app.services.mappers import SinkMapper, SourceMapper
from app.services.parent_process_guard import parent_process_only
from app.utils import Singleton

logger = logging.getLogger(__name__)


class PipelineField(StrEnum):
    """Enumeration for pipeline fields that can trigger configuration reloads."""

    SOURCE_ID = "source_id"
    SINK_ID = "sink_id"


class ConfigurationService(metaclass=Singleton):
    def __init__(self) -> None:
        self.source_service: GenericPersistenceService[Source, SourceDB, SourceRepository] = GenericPersistenceService(
            ServiceConfig(SourceRepository, SourceMapper, ResourceType.SOURCE)
        )
        self.sink_service: GenericPersistenceService[Sink, SinkDB, SinkRepository] = GenericPersistenceService(
            ServiceConfig(SinkRepository, SinkMapper, ResourceType.SINK)
        )

    @staticmethod
    def _notify_sink_changed() -> None:
        ActivePipelineService().reload()

    @staticmethod
    def _notify_source_changed() -> None:
        from app.core import Scheduler  # noqa: PLC0415 # cyclic-import

        condition = Scheduler().mp_config_changed_condition
        with condition:
            condition.notify_all()

    @staticmethod
    def _on_config_changed(config_id: UUID, field: PipelineField, db: Session, notify_fn: Callable[[], None]) -> None:
        """Notify threads or child processes that the configuration has changed.
        Notification triggered only when the configuration is used by the active pipeline."""
        pipeline_repo = PipelineRepository(db)
        active_pipeline = pipeline_repo.get_active_pipeline()
        if active_pipeline and getattr(active_pipeline, field) == str(config_id):
            notify_fn()

    def list_sources(self) -> list[Source]:
        return self.source_service.list_all()

    def list_sinks(self) -> list[Sink]:
        return self.sink_service.list_all()

    def get_source_by_id(self, source_id: UUID, db: Session | None = None) -> Source:
        source = self.source_service.get_by_id(source_id, db)
        if not source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        return source

    def get_sink_by_id(self, sink_id: UUID, db: Session | None = None) -> Sink:
        sink = self.sink_service.get_by_id(sink_id, db)
        if not sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        return sink

    @parent_process_only
    def create_source(self, source: Source) -> Source:
        return self.source_service.create(source)

    @parent_process_only
    def create_sink(self, sink: Sink) -> Sink:
        return self.sink_service.create(sink)

    @parent_process_only
    def update_source(self, source_id: UUID, partial_config: dict) -> Source:
        with get_db_session() as db:
            source = self.get_source_by_id(source_id, db)
            updated = self.source_service.update(source, partial_config, db)
            db.commit()
            self._on_config_changed(updated.id, PipelineField.SOURCE_ID, db, self._notify_source_changed)
            return updated

    @parent_process_only
    def update_sink(self, sink_id: UUID, partial_config: dict) -> Sink:
        with get_db_session() as db:
            sink = self.get_sink_by_id(sink_id, db)
            updated = self.sink_service.update(sink, partial_config, db)
            db.commit()
            self._on_config_changed(updated.id, PipelineField.SINK_ID, db, self._notify_sink_changed)
            return updated

    @parent_process_only
    def delete_source_by_id(self, source_id: UUID) -> None:
        with get_db_session() as db:
            source = self.get_source_by_id(source_id, db)
            self.source_service.delete_by_id(source.id, db)
            db.commit()

    @parent_process_only
    def delete_sink_by_id(self, sink_id: UUID) -> None:
        with get_db_session() as db:
            sink = self.get_sink_by_id(sink_id, db)
            self.sink_service.delete_by_id(sink.id, db)
            db.commit()
