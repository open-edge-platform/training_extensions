# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.repositories.base import BaseRepository


class LabelRepository(BaseRepository[LabelDB]):
    """Repository for label-related database operations."""

    def __init__(self, project_id: str, db: Session):
        super().__init__(db, LabelDB)
        self.project_id = project_id

    def list_all(self) -> list[LabelDB]:
        """Get labels by project ID."""
        return self.db.query(LabelDB).filter(LabelDB.project_id == self.project_id).all()

    def delete_batch(self, obj_ids: list[str]) -> None:
        """Delete labels by IDs within the project."""
        self.db.query(self.model).filter(
            self.model.id.in_(obj_ids),
            self.model.project_id == self.project_id,
        ).delete(synchronize_session=False)
        self.db.flush()
