# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import StrEnum
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import SinkRepository, SourceRepository
from app.repositories.base import UniqueConstraintIntegrityError
from app.schemas import Sink, Source

from .base import (
    GenericPersistenceService,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithNameAlreadyExistsError,
    ServiceConfig,
)
from .event.event_bus import EventBus, EventType
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

    def get_active_source(self) -> Source | None:
        with self._get_repo() as repo:
            item_db = repo.get_active_source()
            return self.config.mapper_class.to_schema(item_db) if item_db else None


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

    def get_active_sink(self) -> Sink | None:
        with self._get_repo() as repo:
            item_db = repo.get_active_sink()
            return self.config.mapper_class.to_schema(item_db) if item_db else None


class ConfigurationService:
    def __init__(self, event_bus: EventBus, db_session: Session) -> None:
        self._event_bus: EventBus = event_bus
        self._source_service: SourceService = SourceService(db_session)
        self._sink_service: SinkService = SinkService(db_session)
        self._db_session = db_session

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

    def get_active_source(self) -> Source | None:
        return self._source_service.get_active_source()

    def get_active_sink(self) -> Sink | None:
        return self._sink_service.get_active_sink()

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
        active_source = self._source_service.get_active_source()
        if active_source and active_source.id == updated.id:
            self._event_bus.emit_event(EventType.SOURCE_CHANGED)
        return updated

    @parent_process_only
    def update_sink(self, sink_id: UUID, partial_config: dict) -> Sink:
        sink = self.get_sink_by_id(sink_id)
        updated = self._sink_service.update(sink, partial_config)
        active_sink = self._sink_service.get_active_sink()
        if active_sink and active_sink.id == updated.id:
            self._event_bus.emit_event(EventType.SINK_CHANGED)
        return updated

    @parent_process_only
    def delete_source_by_id(self, source_id: UUID) -> None:
        source = self.get_source_by_id(source_id)
        self._source_service.delete_by_id(source.id)

    @parent_process_only
    def delete_sink_by_id(self, sink_id: UUID) -> None:
        sink = self.get_sink_by_id(sink_id)
        self._sink_service.delete_by_id(sink.id)
