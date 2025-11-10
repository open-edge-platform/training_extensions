# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import SinkDB
from app.models import OutputFormat, Sink, SinkAdapter, SinkType
from app.models.sink import SinkConfig
from app.repositories import SinkRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError

from .base import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
)
from .event.event_bus import EventBus, EventType
from .parent_process_guard import parent_process_only


class SinkService:
    def __init__(self, event_bus: EventBus, db_session: Session):
        self._event_bus: EventBus = event_bus
        self._db_session = db_session

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

    @parent_process_only
    def update_sink(
        self,
        sink: Sink,
        new_name: str,
        new_rate_limit: float | None,
        new_config_data: SinkConfig,
        new_output_formats: list[OutputFormat],
    ) -> Sink:
        try:
            sink_repo = SinkRepository(self._db_session)
            db_sink = sink_repo.update(
                SinkDB(
                    id=str(sink.id),
                    name=new_name,
                    rate_limit=new_rate_limit,
                    config_data=new_config_data.model_dump(mode="json"),
                    output_formats=new_output_formats,
                )
            )
            active_sink_id = self.get_active_sink_id()
            if active_sink_id == UUID(db_sink.id):
                self._event_bus.emit_event(EventType.SINK_CHANGED)
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

    @parent_process_only
    def delete_sink(self, sink: Sink) -> None:
        try:
            deleted = SinkRepository(self._db_session).delete(str(sink.id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.SINK, str(sink.id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SINK, str(sink.id))

    def get_active_sink(self) -> Sink | None:
        db_sink = SinkRepository(self._db_session).get_active_sink()
        return SinkAdapter.validate_python(db_sink, from_attributes=True) if db_sink else None

    def get_active_sink_id(self) -> UUID | None:
        id = SinkRepository(self._db_session).get_active_sink_id()
        return UUID(id) if id else None
