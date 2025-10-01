# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from sqlite3 import IntegrityError as SQLIntegrityError
from typing import TypeVar

from sqlalchemy import exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import Base

ModelType = TypeVar("ModelType", bound=Base)


class PrimaryKeyIntegrityError(Exception):
    pass


class UniqueConstraintIntegrityError(Exception):
    pass


class BaseRepository[ModelType]:
    """Base repository class for database operations."""

    def __init__(self, db: Session, model: type[ModelType]) -> None:
        self.db = db
        self.model = model

    def get_by_id(self, obj_id: str) -> ModelType | None:
        return self.db.get(self.model, obj_id)

    def exists(self, obj_id: str) -> bool:
        return self.db.query(exists().where(self.model.id == obj_id)).scalar()  # type: ignore[attr-defined]

    def list_all(self) -> list[ModelType]:
        return self.db.query(self.model).all()

    def save(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now()  # type: ignore[attr-defined]
        self.db.add(item)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        return item

    def update(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now()  # type: ignore[attr-defined]
        updated = self.db.merge(item)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        self.db.refresh(updated)
        return updated

    def delete(self, obj_id: str) -> bool:
        return self.db.query(self.model).filter(self.model.id == obj_id).delete() > 0  # type: ignore[attr-defined]

    def save_batch(self, items: list[ModelType]) -> list[ModelType]:
        for item in items:
            item.updated_at = datetime.now()  # type: ignore[attr-defined]
        self.db.add_all(items)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)
        return items

    def update_batch(self, updates: list[ModelType]) -> None:
        for update in updates:
            update.updated_at = datetime.now()  # type: ignore[attr-defined]
            self.db.merge(update)
        try:
            self.db.flush()
        except IntegrityError as e:
            BaseRepository._handle_integrity_error(e)

    def delete_batch(self, obj_ids: list[str]) -> None:
        self.db.query(self.model).filter(self.model.id.in_(obj_ids)).delete(synchronize_session=False)  # type: ignore[attr-defined]
        self.db.flush()

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
