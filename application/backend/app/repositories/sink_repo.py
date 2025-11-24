# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB, SinkDB
from app.repositories.base import BaseRepository


class SinkRepository(BaseRepository[SinkDB]):
    """Repository for sink-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, SinkDB)

    def get_active_sink(self) -> SinkDB | None:
        """Retrieve a sink of an active pipeline."""
        stmt = select(SinkDB).join(PipelineDB, SinkDB.id == PipelineDB.sink_id).where(PipelineDB.is_running)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_active_sink_id(self) -> str | None:
        """Retrieve a sink ID of an active pipeline."""
        stmt = select(SinkDB.id).join(PipelineDB, SinkDB.id == PipelineDB.sink_id).where(PipelineDB.is_running)
        return self.db.execute(stmt).scalar_one_or_none()
