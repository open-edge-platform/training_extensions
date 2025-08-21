# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

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

    def update_source(self, pipeline_id: str, source_id: str) -> None:
        """Update pipeline's source."""
        pipeline = self.get_by_id(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        pipeline.source_id = source_id
        pipeline.updated_at = datetime.now()

    def update_sink(self, pipeline_id: str, sink_id: str) -> None:
        """Update pipeline's sink."""
        pipeline = self.get_by_id(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        pipeline.sink_id = sink_id
        pipeline.updated_at = datetime.now()
