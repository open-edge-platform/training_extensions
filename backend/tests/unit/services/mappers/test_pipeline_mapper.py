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

SUPPORTED_PIPELINE_MAPPING = [
    (
        Pipeline(project_id=UUID0, status=PipelineStatus.IDLE),
        PipelineDB(project_id=str(UUID0), is_running=False),
    ),
    (
        Pipeline(
            project_id=UUID0,
            status=PipelineStatus.RUNNING,
            source_id=UUID1,
            sink_id=UUID2,
            model_id=UUID3,
        ),
        PipelineDB(
            project_id=str(UUID0),
            is_running=True,
            source_id=str(UUID1),
            sink_id=str(UUID2),
            model_id=str(UUID3),
        ),
    ),
]


class TestPipelineMapper:
    """Test suite for PipelineMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", SUPPORTED_PIPELINE_MAPPING.copy())
    def test_from_schema(self, schema_instance, expected_db):
        actual_db = PipelineMapper.from_schema(schema_instance)
        assert actual_db.project_id == expected_db.project_id
        assert actual_db.is_running == expected_db.is_running
        assert actual_db.source_id == expected_db.source_id
        assert actual_db.sink_id == expected_db.sink_id
        assert actual_db.model_id == expected_db.model_id

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_PIPELINE_MAPPING.copy()])
    def test_to_schema(self, db_instance, expected_schema):
        actual_schema = PipelineMapper.to_schema(db_instance)
        assert actual_schema == expected_schema

    def test_misconfigured_pipeline(self):
        with pytest.raises(ValidationError):
            PipelineMapper.to_schema(PipelineDB(project_id=str(UUID0), is_running=True))

        with pytest.raises(ValidationError):
            Pipeline(project_id=UUID1, status=PipelineStatus.RUNNING)
