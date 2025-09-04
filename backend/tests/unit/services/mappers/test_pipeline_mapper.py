# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from pydantic_core._pydantic_core import ValidationError

from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus
from app.services.mappers import PipelineMapper

UUID0 = uuid4()
UUID1 = uuid4()
UUID2 = uuid4()
UUID3 = uuid4()


class TestPipelineMapper:
    """Test suite for PipelineMapper methods."""

    @pytest.mark.parametrize(
        "schema_instance,expected_db",
        [
            (
                PipelineDB(project_id=str(UUID0), is_running=False),
                Pipeline(project_id=UUID0, status=PipelineStatus.IDLE),
            ),
            # (
            #     Pipeline(
            #         project_id=UUID0,
            #         status=PipelineStatus.RUNNING,
            #         source_id=UUID1,
            #         sink_id=UUID2,
            #         model_id=UUID3,
            #     ),
            #     PipelineDB(
            #         project_id=str(UUID0),
            #         is_running=True,
            #         source_id=str(UUID1),
            #         sink_id=str(UUID2),
            #         model_id=str(UUID3),
            #     ),
            # ),
        ],
    )
    def test_from_schema(self, schema_instance, expected_db):
        pipeline_id = uuid4()
        schema_instance.id = pipeline_id
        expected_db.id = str(pipeline_id)
        result = PipelineMapper.from_schema(schema_instance)
        assert result.is_running == expected_db.is_running
        assert result.source_id == expected_db.source_id
        assert result.sink_id == expected_db.sink_id
        assert result.model_id == expected_db.model_id

    @pytest.mark.parametrize(
        "db_instance,expected_schema",
        [
            (
                PipelineDB(project_id=str(UUID0), is_running=False),
                Pipeline(project_id=UUID0, status=PipelineStatus.IDLE),
            ),
            # (
            #     PipelineDB(
            #         project_id=str(UUID0),
            #         is_running=True,
            #         source_id=str(UUID1),
            #         sink_id=str(UUID2),
            #         model_id=str(UUID3),
            #     ),
            #     Pipeline(
            #         project_id=UUID0,
            #         status=PipelineStatus.RUNNING,
            #         source=UUID1,
            #         sink=UUID2,
            #         model=UUID3,
            #     ),
            # ),
        ],
    )
    def test_to_schema(self, db_instance, expected_schema):
        pipeline_id = uuid4()
        db_instance.id = str(pipeline_id)
        expected_schema.id = pipeline_id
        result = PipelineMapper.to_schema(db_instance)
        assert result == expected_schema

    def test_misconfigured_pipeline(self):
        db_instance = PipelineDB(
            project_id=str(UUID1),
            is_running=True,
        )
        with pytest.raises(ValidationError):
            PipelineMapper.to_schema(db_instance)

        with pytest.raises(ValidationError):
            Pipeline(
                project_id=UUID1,
                status=PipelineStatus.RUNNING,
            )
