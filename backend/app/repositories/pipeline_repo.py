# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.repositories.base import BaseRepository


class PipelineRepository(BaseRepository[PipelineDB]):
    """Repository for pipeline-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, PipelineDB)

    def get_active_pipeline(self) -> PipelineDB | None:
        """Get the active pipeline from database."""
        return self.db.query(PipelineDB).filter(PipelineDB.is_running).first()
