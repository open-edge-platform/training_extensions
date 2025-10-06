# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, PipelineDB
from app.repositories.base import BaseRepository


class ModelRevisionRepository(BaseRepository[ModelRevisionDB]):
    """Repository for model-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ModelRevisionDB)

    def get_active_revision(self) -> ModelRevisionDB | None:
        """
        Get the active model revision from database.

        An active model revision is one that is associated with a running pipeline.
        """
        stmt = (
            select(ModelRevisionDB)
            .join(
                PipelineDB,
                (ModelRevisionDB.id == PipelineDB.model_revision_id)
                & (ModelRevisionDB.project_id == PipelineDB.project_id),
            )
            .where(PipelineDB.is_running)
        )
        return self.db.execute(stmt).scalar_one_or_none()
