# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from enum import StrEnum
from multiprocessing.synchronize import Condition
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import PipelineRepository, SinkRepository, SourceRepository
from app.repositories.base import UniqueConstraintIntegrityError
from app.schemas import Sink, Source

from .active_pipeline_service import ActivePipelineService
from .base import (
    GenericPersistenceService,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithNameAlreadyExistsError,
    ServiceConfig,
)
from .mappers import SinkMapper, SourceMapper
from .parent_process_guard import parent_process_only

logger = logging.getLogger(__name__)


class PipelineField(StrEnum):
    """Enumeration for pipeline fields that can trigger configuration reloads."""

    SOURCE_ID = "source_id"
    SINK_ID = "sink_id"


class SourceService(GenericPersistenceService[Source, SourceRepository]):
    def __init__(self, db_session: Session):
        super().__init__(ServiceConfig(SourceRepository, SourceMapper, ResourceType.SOURCE), db_session)

    def create(self, item: Source) -> Source:
        try:
            return super().create(item)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, item.name)

    def update(self, item: Source, partial_config: dict) -> Source:
        try:
            return super().update(item, partial_config)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, partial_config["name"])


class SinkService(GenericPersistenceService[Sink, SinkRepository]):
    def __init__(self, db_session: Session):
        super().__init__(ServiceConfig(SinkRepository, SinkMapper, ResourceType.SINK), db_session)

    def create(self, item: Sink) -> Sink:
        try:
            return super().create(item)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, item.name)

    def update(self, item: Sink, partial_config: dict) -> Sink:
        try:
            return super().update(item, partial_config)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, partial_config["name"])


class ConfigurationService:
    def __init__(
        self, active_pipeline_service: ActivePipelineService, db_session: Session, config_changed_condition: Condition
    ) -> None:
        self._source_service: SourceService = SourceService(db_session)
        self._sink_service: SinkService = SinkService(db_session)
        self._db_session = db_session
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._config_changed_condition: Condition = config_changed_condition

    def _notify_sink_changed(self) -> None:
        self._active_pipeline_service.reload()

    def _notify_source_changed(self) -> None:
        with self._config_changed_condition:
            self._config_changed_condition.notify_all()

    def _on_config_changed(self, config_id: UUID, field: PipelineField, notify_fn: Callable[[], None]) -> None:
        """Notify threads or child processes that the configuration has changed.
        Notification triggered only when the configuration is used by the active pipeline."""
        pipeline_repo = PipelineRepository(self._db_session)
        active_pipeline = pipeline_repo.get_active_pipeline()
        if active_pipeline and getattr(active_pipeline, field) == str(config_id):
            notify_fn()

    def list_sources(self) -> list[Source]:
        return self._source_service.list_all()

    def list_sinks(self) -> list[Sink]:
        return self._sink_service.list_all()

    def get_source_by_id(self, source_id: UUID) -> Source:
        source = self._source_service.get_by_id(source_id)
        if not source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        return source

    def get_sink_by_id(self, sink_id: UUID) -> Sink:
        sink = self._sink_service.get_by_id(sink_id)
        if not sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        return sink

    @parent_process_only
    def create_source(self, source: Source) -> Source:
        return self._source_service.create(source)

    @parent_process_only
    def create_sink(self, sink: Sink) -> Sink:
        return self._sink_service.create(sink)

    @parent_process_only
    def update_source(self, source_id: UUID, partial_config: dict) -> Source:
        source = self.get_source_by_id(source_id)
        updated = self._source_service.update(source, partial_config)
        self._on_config_changed(updated.id, PipelineField.SOURCE_ID, self._notify_source_changed)
        return updated

    @parent_process_only
    def update_sink(self, sink_id: UUID, partial_config: dict) -> Sink:
        sink = self.get_sink_by_id(sink_id)
        updated = self._sink_service.update(sink, partial_config)
        self._on_config_changed(updated.id, PipelineField.SINK_ID, self._notify_sink_changed)
        return updated

    @parent_process_only
    def delete_source_by_id(self, source_id: UUID) -> None:
        source = self.get_source_by_id(source_id)
        self._source_service.delete_by_id(source.id)

    @parent_process_only
    def delete_sink_by_id(self, sink_id: UUID) -> None:
        sink = self.get_sink_by_id(sink_id)
        self._sink_service.delete_by_id(sink.id)
