# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import TypeVar

from sqlalchemy.orm import Session

from app.db.schema import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository[ModelType]:
    """Base repository class for database operations."""

    def __init__(self, db: Session, model: type[ModelType]):
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
        self.db.merge(item)
        self.db.flush()
        return item

    def delete(self, obj_id: str) -> None:
        self.db.query(self.model).filter(self.model.id == obj_id).delete()  # type: ignore[attr-defined]
