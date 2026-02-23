# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetRevisionDB
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

    def get_latest_uptodate_dataset_revision(self) -> DatasetRevisionDB | None:
        """
        Get latest up to date created dataset revision, if it exists.

        Up to date means the dataset was created after the last update on any dataset item in the project.
        """
        stmt = (
            select(DatasetItemDB)
            .where(DatasetItemDB.project_id == self.project_id)
            .order_by(DatasetItemDB.updated_at.desc())
            .limit(1)
        )
        latest_updated_dataset_item = self.db.execute(stmt).scalar_one_or_none()
        if latest_updated_dataset_item is None:
            return None

        stmt = (
            select(DatasetRevisionDB)
            .where(
                (DatasetRevisionDB.project_id == self.project_id)
                & (DatasetRevisionDB.created_at > latest_updated_dataset_item.updated_at)
            )
            .order_by(DatasetRevisionDB.created_at.desc())
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def delete(self, obj_id: str) -> bool:
        """Delete a dataset revision by its id."""
        stmt = delete(DatasetRevisionDB).where(
            (DatasetRevisionDB.id == obj_id) & (DatasetRevisionDB.project_id == self.project_id)
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0
