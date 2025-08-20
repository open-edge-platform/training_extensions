# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from app.schemas import SinkType, SourceType
from app.services.active_pipeline_service import ActivePipelineService


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.active_pipeline_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock


class TestActivePipelineServiceIntegration:
    """Integration tests for ActivePipelineService."""

    def test_load_default_config(self):
        """Test default configuration settings."""
        active_pipeline_service = ActivePipelineService()
        source = active_pipeline_service.get_source_config()
        sink = active_pipeline_service.get_sink_config()

        assert source.source_type == SourceType.DISCONNECTED
        assert sink.sink_type == SinkType.DISCONNECTED
