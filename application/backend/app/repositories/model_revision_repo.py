# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select, update
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

    def update_training_status(
        self,
        obj_id: str,
        training_status: str,
        training_started_at: datetime | None = None,
        training_finished_at: datetime | None = None,
    ) -> None:
        """
        Update the training status and trainig start and finish time of a model revision.

        Args:
            obj_id (str): Unique identifier of the model revision to update.
            training_status (str): New training status value to set.
            training_started_at (datetime): Date and time when the training was started
            training_finished_at (datetime): Date and time when the training was finished
        """
        values: dict[str, Any] = {"training_status": training_status}
        if training_started_at is not None:
            values["training_started_at"] = training_started_at
        if training_finished_at is not None:
            values["training_finished_at"] = training_finished_at

        stmt = (
            update(ModelRevisionDB)
            .where((ModelRevisionDB.id == obj_id) & (ModelRevisionDB.project_id == self.project_id))
            .values(**values)
        )
        self.db.execute(stmt)
