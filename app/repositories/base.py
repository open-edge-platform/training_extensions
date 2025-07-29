from datetime import datetime
from typing import Generic, TypeVar

from sqlalchemy.orm import Session

from app.db.schema import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository class for database operations."""

    def __init__(self, db: Session, model: type[ModelType]):
        self.db = db
        self.model = model

    def get_by_id(self, obj_id: str) -> ModelType | None:
        return self.db.query(self.model).filter(self.model.id == obj_id).first()

    def list_all(self) -> list[ModelType]:
        return self.db.query(self.model).all()

    def save(self, item: ModelType) -> ModelType:
        item.updated_at = datetime.now()
        self.db.add(item)
        self.db.flush()
        return item
