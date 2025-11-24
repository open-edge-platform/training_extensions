# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB


class PipelineRepository:
    """Repository for pipeline-related database operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, obj_id: str) -> PipelineDB | None:
        return self.db.get(PipelineDB, obj_id)

    def save(self, item: PipelineDB) -> PipelineDB:
        self.db.add(item)
        self.db.flush()
        return item

    def is_running(self, obj_id: str) -> bool:
        stmt = select(PipelineDB.is_running).select_from(PipelineDB).where(PipelineDB.project_id == obj_id)
        return self.db.scalar(stmt) or False

    def update(self, item: PipelineDB) -> PipelineDB:
        item.updated_at = datetime.now(UTC)
        updated = self.db.merge(item)
        # Explicit early commit to ensure changes are visible to other transactions before the session is closed
        self.db.commit()
        return updated

    def get_active_pipeline(self) -> PipelineDB | None:
        """Get the active pipeline from database."""
        stmt = select(PipelineDB).where(PipelineDB.is_running)
        return self.db.execute(stmt).scalar_one_or_none()
