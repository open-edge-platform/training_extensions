# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB, SourceDB
from app.repositories.base import BaseRepository


class SourceRepository(BaseRepository[SourceDB]):
    """Repository for source-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, SourceDB)

    def get_active_source(self) -> SourceDB | None:
        """Retrieve a source of an active pipeline."""
        stmt = select(SourceDB).join(PipelineDB, SourceDB.id == PipelineDB.source_id).where(PipelineDB.is_running)
        return self.db.execute(stmt).scalar_one_or_none()
