# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, PipelineDB
from app.repositories.base import BaseRepository


class ModelRevisionRepository(BaseRepository[ModelRevisionDB]):
    """Repository for model-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ModelRevisionDB)

    def __get_active_pipeline(self) -> PipelineDB | None:
        """Get the active pipeline from database."""
        return self.db.query(PipelineDB).filter(PipelineDB.is_running).first()

    def get_active_revision(self) -> ModelRevisionDB | None:
        pipeline = self.__get_active_pipeline()
        if not pipeline:
            return None
        return self.db.query(ModelRevisionDB).filter(ModelRevisionDB.id == pipeline.model_revision_id).first()
