# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from pydantic_core._pydantic_core import ValidationError

from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus
from app.services.mappers import PipelineMapper

UUID1 = uuid4()
UUID2 = uuid4()
UUID3 = uuid4()


class TestPipelineMapper:
    """Test suite for PipelineMapper methods."""

    @pytest.mark.parametrize(
        "db_instance,expected_schema",
        [
            (
                PipelineDB(name="Idle Pipeline", description="A test pipeline", is_running=False),
                Pipeline(name="Idle Pipeline", status=PipelineStatus.IDLE),
            ),
            (
                PipelineDB(
                    name="Running Pipeline",
                    description="A test pipeline",
                    is_running=True,
                    source_id=str(UUID1),
                    sink_id=str(UUID2),
                    model_id=str(UUID3),
                ),
                Pipeline(
                    name="Running Pipeline",
                    status=PipelineStatus.RUNNING,
                    source_id=UUID1,
                    sink_id=UUID2,
                    model_id=UUID3,
                ),
            ),
        ],
    )
    def test_to_schema(self, db_instance, expected_schema):
        pipeline_id = uuid4()
        db_instance.id = str(pipeline_id)
        expected_schema.id = pipeline_id
        result = PipelineMapper.to_schema(db_instance)
        assert result == expected_schema

    @pytest.mark.parametrize(
        "schema_instance,expected_db",
        [
            (
                Pipeline(name="Idle Pipeline", status=PipelineStatus.IDLE),
                PipelineDB(name="Idle Pipeline", description="A test pipeline", is_running=False),
            ),
            (
                Pipeline(
                    name="Running Pipeline",
                    status=PipelineStatus.RUNNING,
                    source_id=UUID1,
                    sink_id=UUID2,
                    model_id=UUID3,
                ),
                PipelineDB(
                    name="Running Pipeline",
                    description="A test pipeline",
                    is_running=True,
                    source_id=str(UUID1),
                    sink_id=str(UUID2),
                    model_id=str(UUID3),
                ),
            ),
        ],
    )
    def test_from_schema(self, schema_instance, expected_db):
        pipeline_id = uuid4()
        schema_instance.id = pipeline_id
        expected_db.id = str(pipeline_id)
        result = PipelineMapper.from_schema(schema_instance)
        assert result.id == expected_db.id
        assert result.name == expected_db.name
        assert result.is_running == expected_db.is_running
        assert result.source_id == expected_db.source_id
        assert result.sink_id == expected_db.sink_id
        assert result.model_id == expected_db.model_id

    def test_misconfigured_pipeline(self):
        db_instance = PipelineDB(
            id=str(uuid4()),
            name="Misconfigured Pipeline",
            is_running=True,
        )
        with pytest.raises(ValidationError):
            PipelineMapper.to_schema(db_instance)

        with pytest.raises(ValidationError):
            Pipeline(
                id=uuid4(),
                name="Misconfigured Pipeline",
                status=PipelineStatus.RUNNING,
            )
