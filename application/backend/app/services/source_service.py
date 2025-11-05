# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import SourceRepository
from app.repositories.base import UniqueConstraintIntegrityError
from app.schemas import Source

from .base import (
    GenericPersistenceService,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithNameAlreadyExistsError,
    ServiceConfig,
)
from .event.event_bus import EventBus, EventType
from .mappers import SourceMapper
from .parent_process_guard import parent_process_only


class SourceService(GenericPersistenceService[Source, SourceRepository]):
    def __init__(self, db_session: Session):
        super().__init__(ServiceConfig(SourceRepository, SourceMapper, ResourceType.SOURCE), db_session)

    def get_by_id(self, item_id: UUID) -> Source:
        source = super().get_by_id(item_id)
        if not source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(item_id))
        return source

    @parent_process_only
    def create(self, item: Source) -> Source:
        try:
            return super().create(item)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, item.name)

    def get_active_source(self) -> Source | None:
        with self._get_repo() as repo:
            item_db = repo.get_active_source()
            return self.config.mapper_class.to_schema(item_db) if item_db else None

    @parent_process_only
    def delete_by_id(self, item_id: UUID) -> None:
        super().delete_by_id(item_id)


class SourceUpdateService(SourceService):
    def __init__(self, event_bus: EventBus, db_session: Session):
        self._event_bus: EventBus = event_bus
        super().__init__(db_session)

    @parent_process_only
    def update(self, source: Source, partial_config: dict) -> Source:
        try:
            updated = super().update(source, partial_config)
            active_source = self.get_active_source()
            if active_source and active_source.id == updated.id:
                self._event_bus.emit_event(EventType.SOURCE_CHANGED)
            return updated
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SOURCE, partial_config["name"])
