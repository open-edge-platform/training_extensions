# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import TypeVar

from sqlalchemy.orm import Session

from app.db.schema import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository[ModelType]:
    """Base repository class for database operations."""

    def __init__(self, db: Session, model: type[ModelType]) -> None:
        self.db = db
        self.model = model

    def get_by_id(self, obj_id: str) -> ModelType | None:
        return self.db.get(self.model, obj_id)

    def list_all(self) -> list[ModelType]:
        return self.db.query(self.model).all()

    def save(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now()  # type: ignore[attr-defined]
        self.db.add(item)
        self.db.flush()
        return item

    def update(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now()  # type: ignore[attr-defined]
        updated = self.db.merge(item)
        self.db.flush()
        self.db.refresh(updated)
        return updated

    def delete(self, obj_id: str) -> bool:
        return self.db.query(self.model).filter(self.model.id == obj_id).delete() > 0  # type: ignore[attr-defined]

    def save_batch(self, items: list[ModelType]) -> list[ModelType]:
        for item in items:
            item.updated_at = datetime.now()  # type: ignore[attr-defined]
        self.db.add_all(items)
        self.db.flush()
        return items

    def update_batch(self, updates: list[ModelType]) -> None:
        for update in updates:
            update.updated_at = datetime.now()  # type: ignore[attr-defined]
            self.db.merge(update)
        self.db.flush()

    def delete_batch(self, obj_ids: list[str]) -> None:
        self.db.query(self.model).filter(self.model.id.in_(obj_ids)).delete(synchronize_session=False)  # type: ignore[attr-defined]
        self.db.flush()
