# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.repositories.base import BaseRepository


class LabelRepository(BaseRepository[LabelDB]):
    """Repository for label-related database operations."""

    def __init__(self, project_id: str, db: Session):
        super().__init__(db, LabelDB)
        self.project_id = project_id

    def list_all(self) -> Sequence[LabelDB]:
        """Get labels by project ID."""
        stmt = select(LabelDB).where(LabelDB.project_id == self.project_id)
        return self.db.execute(stmt).scalars().all()

    def list_ids(self) -> Sequence[str]:
        """Get labels ID's by project ID."""
        stmt = select(LabelDB.id).where(LabelDB.project_id == self.project_id)
        return self.db.execute(stmt).scalars().all()

    def delete_batch(self, obj_ids: list[str]) -> int:
        """Delete labels by IDs within the project."""
        stmt = delete(self.model).where((self.model.id.in_(obj_ids)) & (self.model.project_id == self.project_id))
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount
