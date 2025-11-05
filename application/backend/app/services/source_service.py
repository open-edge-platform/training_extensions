# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import SourceDB
from app.models import Source, SourceType
from app.models.source import SourceAdapter, SourceConfig
from app.repositories import SourceRepository
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


class SourceService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    @parent_process_only
    def create_source(
        self,
        name: str,
        source_type: SourceType,
        config_data: SourceConfig,
        source_id: UUID | None = None,
    ) -> Source:
        try:
            db_source = SourceRepository(self._db_session).save(
                SourceDB(
                    id=str(source_id) if source_id is not None else None,
                    name=name,
                    source_type=source_type,
                    config_data=config_data.model_dump(mode="json"),
                )
            )
            return SourceAdapter.validate_python(db_source, from_attributes=True)
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.SOURCE, str(source_id))
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, name)

    def get_by_id(self, source_id: UUID) -> Source:
        db_source = SourceRepository(self._db_session).get_by_id(str(source_id))
        if not db_source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        return SourceAdapter.validate_python(db_source, from_attributes=True)

    def list_all(self) -> list[Source]:
        return [
            SourceAdapter.validate_python(db_source, from_attributes=True)
            for db_source in SourceRepository(self._db_session).list_all()
        ]

    @parent_process_only
    def delete_source(self, source: Source) -> None:
        try:
            deleted = SourceRepository(self._db_session).delete(str(source.id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source.id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SOURCE, str(source.id))

    def get_active_source(self) -> Source | None:
        db_source = SourceRepository(self._db_session).get_active_source()
        return SourceAdapter.validate_python(db_source, from_attributes=True) if db_source else None

    def get_active_source_id(self) -> UUID | None:
        id = SourceRepository(self._db_session).get_active_source_id()
        return UUID(id) if id else None


class SourceUpdateService(SourceService):
    def __init__(self, event_bus: EventBus, db_session: Session):
        self._event_bus: EventBus = event_bus
        super().__init__(db_session)

    @parent_process_only
    def update_source(
        self,
        source: Source,
        new_name: str,
        new_config_data: SourceConfig,
    ) -> Source:
        try:
            source_repo = SourceRepository(self._db_session)
            db_source = source_repo.update(
                SourceDB(
                    id=str(source.id),
                    name=new_name,
                    config_data=new_config_data.model_dump(mode="json"),
                )
            )
            active_source_id = self.get_active_source_id()
            if active_source_id == UUID(db_source.id):
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            return SourceAdapter.validate_python(db_source, from_attributes=True)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, new_name)
