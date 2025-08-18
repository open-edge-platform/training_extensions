from datetime import datetime

from sqlalchemy.orm import Session

from app.db.schema import ModelDB, PipelineDB
from app.repositories.base import BaseRepository


class ModelRepository(BaseRepository[ModelDB]):
    """Repository for model-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ModelDB)

    def __get_active_pipeline(self) -> PipelineDB | None:
        """Get the active pipeline from database."""
        return self.db.query(PipelineDB).filter(PipelineDB.is_running).first()

    def get_active_model(self) -> ModelDB | None:
        pipeline = self.__get_active_pipeline()
        if not pipeline:
            return None
        return self.db.query(ModelDB).filter(ModelDB.id == pipeline.model_id).first()

    def get_by_name(self, model_name: str) -> ModelDB | None:
        return self.db.query(ModelDB).filter(ModelDB.name == model_name).first()

    def remove(self, model_name: str) -> None:
        model = self.get_by_name(model_name)
        if not model:
            raise ValueError(f"Model with name '{model_name}' not found")
        self.db.delete(model)
        self.db.flush()

    def set_active_model(self, model_name: str) -> None:
        pipeline = self.__get_active_pipeline()
        if not pipeline:
            raise ValueError("No active pipeline found")
        model = self.get_by_name(model_name)
        if not model:
            raise ValueError(f"Model with name '{model_name}' not found")
        pipeline.updated_at = datetime.now()
        pipeline.model_id = model.id
