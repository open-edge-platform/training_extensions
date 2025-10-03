# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Generic, TypeVar

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from app.db.schema import BaseID

ModelType = TypeVar("ModelType", bound=BaseID)


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
        self.db.flush()
        return item

    def update(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now(UTC)
        updated = self.db.merge(item)
        self.db.flush()
        self.db.refresh(updated)
        return updated

    def delete(self, obj_id: str) -> bool:
        stmt = delete(self.model).where(self.model.id == obj_id)
        result = self.db.execute(stmt)
        return result.rowcount > 0  # type: ignore[union-attr]

    def save_batch(self, items: list[ModelType]) -> list[ModelType]:
        now = datetime.now(UTC)
        for item in items:
            item.updated_at = now
        self.db.add_all(items)
        self.db.flush()
        return items

    def update_batch(self, updates: list[ModelType]) -> None:
        now = datetime.now(UTC)
        for update in updates:
            update.updated_at = now
            self.db.merge(update)
        self.db.flush()

    def delete_batch(self, obj_ids: list[str]) -> int:
        stmt = delete(self.model).where(self.model.id.in_(obj_ids))
        result = self.db.execute(stmt)
        return result.rowcount  # type: ignore[union-attr]
