# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from datetime import UTC, datetime
from sqlite3 import IntegrityError as SQLIntegrityError
from typing import Generic, TypeVar, cast

from sqlalchemy import CursorResult, delete, exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import BaseID

ModelType = TypeVar("ModelType", bound=BaseID)


class PrimaryKeyIntegrityError(Exception):
    pass


class UniqueConstraintIntegrityError(Exception):
    pass


class BaseRepository(Generic[ModelType]):
    """Base repository class for database operations."""

    def __init__(self, db: Session, model: type[ModelType]) -> None:
        self.db = db
        self.model = model

    def get_by_id(self, obj_id: str) -> ModelType | None:
        return self.db.get(self.model, obj_id)

    def exists(self, obj_id: str) -> bool:
        stmt = select(exists().where(self.model.id == obj_id))
        return self.db.execute(stmt).scalar() or False

    def list_all(self) -> Sequence[ModelType]:
        return self.db.execute(select(self.model)).scalars().all()

    def save(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now(UTC)
        self.db.add(item)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        return item

    def update(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now(UTC)
        updated = self.db.merge(item)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        self.db.refresh(updated)
        return updated

    def delete(self, obj_id: str) -> bool:
        stmt = delete(self.model).where(self.model.id == obj_id)
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

    def save_batch(self, items: list[ModelType]) -> list[ModelType]:
        now = datetime.now(UTC)
        for item in items:
            item.updated_at = now
        self.db.add_all(items)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        return items

    def update_batch(self, updates: list[ModelType]) -> None:
        now = datetime.now(UTC)
        for update in updates:
            update.updated_at = now
            self.db.merge(update)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)

    def delete_batch(self, obj_ids: list[str]) -> int:
        stmt = delete(self.model).where(self.model.id.in_(obj_ids))
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount

    @staticmethod
    def _handle_integrity_error(error: IntegrityError) -> None:
        if error.orig is None or not isinstance(error.orig, SQLIntegrityError):
            raise error
        match error.orig.sqlite_errorname:
            case "SQLITE_CONSTRAINT_PRIMARYKEY":
                raise PrimaryKeyIntegrityError from error
            case "SQLITE_CONSTRAINT_UNIQUE":
                raise UniqueConstraintIntegrityError from error
            case _:
                raise error
