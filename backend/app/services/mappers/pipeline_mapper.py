# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus


class PipelineMapper:
    """Mapper for Pipeline schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(pipeline_db: PipelineDB) -> Pipeline:
        """Convert Pipeline db entity to schema."""
        from app.services.mappers import ModelMapper, SinkMapper, SourceMapper

        return Pipeline(
            project_id=UUID(pipeline_db.project_id),
            source=SourceMapper.to_schema(pipeline_db.source) if pipeline_db.source else None,
            sink=SinkMapper.to_schema(pipeline_db.sink) if pipeline_db.sink else None,
            model=ModelMapper.to_schema(pipeline_db.model) if pipeline_db.model else None,
            sink_id=UUID(pipeline_db.sink_id) if pipeline_db.sink_id else None,
            model_id=UUID(pipeline_db.model_id) if pipeline_db.model_id else None,
            source_id=UUID(pipeline_db.source_id) if pipeline_db.source_id else None,
            status=PipelineStatus.from_bool(pipeline_db.is_running),
        )

    @staticmethod
    def from_schema(pipeline: Pipeline) -> PipelineDB:
        """Convert Pipeline schema to db model."""

        return PipelineDB(
            project_id=str(pipeline.project_id),
            source_id=str(pipeline.source_id) if pipeline.source_id else None,
            model_id=str(pipeline.model_id) if pipeline.model_id else None,
            sink_id=str(pipeline.sink_id) if pipeline.sink_id else None,
            is_running=pipeline.status.as_bool,
        )
