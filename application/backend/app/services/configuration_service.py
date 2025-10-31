# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from enum import StrEnum
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import SinkDB
from app.models import OutputFormat, Sink, SinkAdapter, SinkType
from app.models.sink import SinkConfig
from app.repositories import SinkRepository, SourceRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.schemas import Source

from .base import (
    GenericPersistenceService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
    ServiceConfig,
)
from .event.event_bus import EventBus, EventType
from .mappers import SourceMapper
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


class SinkService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    def create_sink(
        self,
        name: str,
        sink_type: SinkType,
        rate_limit: float | None,
        config_data: SinkConfig,
        output_formats: list[OutputFormat],
        sink_id: UUID | None = None,
    ) -> Sink:
        try:
            db_sink = SinkRepository(self._db_session).save(
                SinkDB(
                    id=str(sink_id) if sink_id is not None else None,
                    name=name,
                    sink_type=sink_type,
                    rate_limit=rate_limit,
                    config_data=config_data.model_dump(mode="json"),
                    output_formats=output_formats,
                )
            )
            return SinkAdapter.validate_python(db_sink, from_attributes=True)
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.SINK, str(sink_id))
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, name)

    def update_sink(
        self,
        sink_id: UUID,
        new_name: str,
        new_rate_limit: float | None,
        new_config_data: SinkConfig,
        new_output_formats: list[OutputFormat],
    ) -> Sink:
        try:
            sink_repo = SinkRepository(self._db_session)
            db_sink = sink_repo.get_by_id(str(sink_id))
            if db_sink is None:
                raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
            db_sink = sink_repo.update(
                SinkDB(
                    id=db_sink.id,
                    name=new_name,
                    rate_limit=new_rate_limit,
                    config_data=new_config_data.model_dump(mode="json"),
                    output_formats=new_output_formats,
                )
            )
            return SinkAdapter.validate_python(db_sink, from_attributes=True)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, new_name)  # type: ignore[arg-type]

    def get_by_id(self, sink_id: UUID) -> Sink:
        db_sink = SinkRepository(self._db_session).get_by_id(str(sink_id))
        if not db_sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        return SinkAdapter.validate_python(db_sink, from_attributes=True)

    def list_all(self) -> list[Sink]:
        return [
            SinkAdapter.validate_python(db_sink, from_attributes=True)
            for db_sink in SinkRepository(self._db_session).list_all()
        ]

    def delete_by_id(self, sink_id: UUID) -> None:
        try:
            deleted = SinkRepository(self._db_session).delete(str(sink_id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SINK, str(sink_id))

    def get_active_sink(self) -> Sink | None:
        db_sink = SinkRepository(self._db_session).get_active_sink()
        return SinkAdapter.validate_python(db_sink, from_attributes=True) if db_sink else None

    def get_active_sink_id(self) -> UUID | None:
        id = SinkRepository(self._db_session).get_active_sink_id()
        return UUID(id) if id else None


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
        return self._sink_service.get_by_id(sink_id)

    def get_active_source(self) -> Source | None:
        return self._source_service.get_active_source()

    def get_active_sink(self) -> Sink | None:
        return self._sink_service.get_active_sink()

    def get_active_sink_id(self) -> UUID | None:
        return self._sink_service.get_active_sink_id()

    @parent_process_only
    def create_source(self, source: Source) -> Source:
        return self._source_service.create(source)

    @parent_process_only
    def create_sink(
        self,
        name: str,
        sink_type: SinkType,
        rate_limit: float | None,
        config_data: SinkConfig,
        output_formats: list[OutputFormat],
        sink_id: UUID | None = None,
    ) -> Sink:
        return self._sink_service.create_sink(
            name=name,
            sink_type=sink_type,
            rate_limit=rate_limit,
            config_data=config_data,
            output_formats=output_formats,
            sink_id=sink_id,
        )

    @parent_process_only
    def update_source(self, source_id: UUID, partial_config: dict) -> Source:
        source = self.get_source_by_id(source_id)
        updated = self._source_service.update(source, partial_config)
        active_source = self._source_service.get_active_source()
        if active_source and active_source.id == updated.id:
            self._event_bus.emit_event(EventType.SOURCE_CHANGED)
        return updated

    @parent_process_only
    def update_sink(
        self,
        sink_id: UUID,
        new_name: str,
        new_rate_limit: float | None,
        new_config_data: SinkConfig,
        new_output_formats: list[OutputFormat],
    ) -> Sink:
        updated = self._sink_service.update_sink(
            sink_id=sink_id,
            new_name=new_name,
            new_rate_limit=new_rate_limit,
            new_config_data=new_config_data,
            new_output_formats=new_output_formats,
        )
        active_sink_id = self._sink_service.get_active_sink_id()
        if active_sink_id == updated.id:
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
