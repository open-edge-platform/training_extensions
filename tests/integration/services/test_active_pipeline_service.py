import multiprocessing as mp
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


@pytest.fixture
def fxt_active_pipeline_service(fxt_default_pipeline) -> ActivePipelineService:
    active_pipeline_service = ActivePipelineService()
    active_pipeline_service.config_changed_condition = mp.Condition()
    active_pipeline_service._load_app_config()
    assert active_pipeline_service._active_pipeline_id == fxt_default_pipeline.id

    return active_pipeline_service


class TestActivePipelineServiceIntegration:
    """Integration tests for ActivePipelineService."""

    def test_load_default_config(self):
        """Test default configuration settings."""

        active_pipeline_service = ActivePipelineService()
        source = active_pipeline_service.get_source_config()
        sink = active_pipeline_service.get_sink_config()

        assert source.source_type == SourceType.DISCONNECTED
        assert sink.sink_type == SinkType.DISCONNECTED

    def test_set_source_success(self, fxt_active_pipeline_service, fxt_source_config):
        """Test setting source configuration successfully."""
        old_source_id = fxt_active_pipeline_service.get_source_config().id
        fxt_active_pipeline_service.set_source_config(fxt_source_config)
        fxt_active_pipeline_service._load_app_config()  # trigger DB loading explicitly

        source_config = fxt_active_pipeline_service.get_source_config()
        assert old_source_id != source_config.id
        assert source_config.device_id == fxt_source_config.device_id

    def test_set_sink_success(self, fxt_active_pipeline_service, fxt_sink_config):
        """Test setting sink configuration successfully."""
        old_sink_id = fxt_active_pipeline_service.get_sink_config().id
        fxt_active_pipeline_service.set_sink_config(fxt_sink_config)
        fxt_active_pipeline_service._load_app_config()  # trigger DB loading explicitly

        sink_config = fxt_active_pipeline_service.get_sink_config()
        assert old_sink_id != sink_config.id
        assert sink_config.rate_limit == fxt_sink_config.rate_limit
        assert sink_config.output_formats == fxt_sink_config.output_formats
        assert sink_config.broker_host == fxt_sink_config.broker_host
        assert sink_config.broker_port == fxt_sink_config.broker_port
        assert sink_config.topic == fxt_sink_config.topic
