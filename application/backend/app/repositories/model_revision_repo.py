# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB
from app.repositories.base import BaseRepository


class ModelRevisionRepository(BaseRepository[ModelRevisionDB]):
    """Repository for model-related database operations."""

    def __init__(self, project_id: str, db: Session):
        super().__init__(db, ModelRevisionDB)
        self.project_id = project_id

    def list_all(self, training_dataset_id: str | None = None) -> Sequence[ModelRevisionDB]:
        """
        List all model revisions for a given project.

        Optionally the model revisions can be filtered on training dataset id

        Args:
            training_dataset_id (str): Optional unique id of the training dataset to filter on

        Returns:
            Sequence[ModelRevisionDB]: A list of model revisions associated with the project.
        """
        stmt = select(ModelRevisionDB).where(ModelRevisionDB.project_id == self.project_id)
        if training_dataset_id is not None:
            stmt = stmt.where(ModelRevisionDB.training_dataset_id == training_dataset_id)
        return self.db.execute(stmt).scalars().all()

    def get_by_id(self, obj_id: str) -> ModelRevisionDB | None:
        """Get the model revision by its id."""
        stmt = select(ModelRevisionDB).where(
            (ModelRevisionDB.id == obj_id) & (ModelRevisionDB.project_id == self.project_id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def delete(self, obj_id: str) -> bool:
        """Delete the model revision by its id."""
        stmt = delete(ModelRevisionDB).where(
            (ModelRevisionDB.id == obj_id) & (ModelRevisionDB.project_id == self.project_id)
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0
