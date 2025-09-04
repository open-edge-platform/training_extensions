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
            source=SourceMapper.to_schema(pipeline_db.source),
            sink=SinkMapper.to_schema(pipeline_db.sink),
            model=ModelMapper.to_schema(pipeline_db.model),
            status=PipelineStatus.from_bool(pipeline_db.is_running),
        )

    @staticmethod
    def from_schema(pipeline: Pipeline) -> PipelineDB:
        """Convert Pipeline schema to db model."""

        return PipelineDB(
            project_id=str(pipeline.project_id),
            source_id=str(pipeline.source.id) if pipeline.source else None,
            model_id=str(pipeline.model.id) if pipeline.model else None,
            sink_id=str(pipeline.sink.id) if pipeline.sink else None,
            is_running=pipeline.status.as_bool,
        )
