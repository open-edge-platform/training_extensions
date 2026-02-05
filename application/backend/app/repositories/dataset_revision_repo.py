# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB
from app.repositories.base import BaseRepository


class DatasetRevisionRepository(BaseRepository[DatasetRevisionDB]):
    """Repository for dataset revision-related database operations."""

    def __init__(self, project_id: str, db: Session):
        super().__init__(db, DatasetRevisionDB)
        self.project_id = project_id

    def list_all(self) -> Sequence[DatasetRevisionDB]:
        """
        List all dataset revisions for a given project.

        Returns:
            Sequence[DatasetRevisionDB]: A list of dataset revisions associated with the project.
        """
        stmt = select(DatasetRevisionDB).where(DatasetRevisionDB.project_id == self.project_id)
        return self.db.execute(stmt).scalars().all()

    def get_by_id(self, obj_id: str) -> DatasetRevisionDB | None:
        """Get a dataset revision by its id."""
        stmt = select(DatasetRevisionDB).where(
            (DatasetRevisionDB.id == obj_id) & (DatasetRevisionDB.project_id == self.project_id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def delete(self, obj_id: str) -> bool:
        """Delete a dataset revision by its id."""
        stmt = delete(DatasetRevisionDB).where(
            (DatasetRevisionDB.id == obj_id) & (DatasetRevisionDB.project_id == self.project_id)
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0
