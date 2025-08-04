import multiprocessing as mp
from unittest.mock import patch

import pytest

from app.schemas import OutputFormat, SinkType, SourceType
from app.schemas.sink import MqttSinkConfig
from app.schemas.source import WebcamSourceConfig
from app.services.configuration_service import ConfigurationService


@pytest.fixture(scope="function", autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.configuration_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock


@pytest.fixture
def fxt_source_config() -> WebcamSourceConfig:
    """Sample source configuration data."""
    return WebcamSourceConfig(source_type=SourceType.WEBCAM, device_id=1)


@pytest.fixture
def fxt_sink_config() -> MqttSinkConfig:
    """Sample sink configuration data."""
    return MqttSinkConfig(
        sink_type=SinkType.MQTT,
        rate_limit=0.1,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        broker_host="localhost",
        broker_port=1883,
        topic="topic",
    )


@pytest.fixture(scope="function")
def fxt_config_service(default_pipeline) -> ConfigurationService:
    config_service = ConfigurationService()
    config_service.config_changed_condition = mp.Condition()
    config_service._load_app_config()
    assert config_service._active_pipeline_id == default_pipeline.id

    return config_service


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_load_app_config_success(self):
        """Test successful loading of application configuration after migration applied."""
        config_service = ConfigurationService()
        source = config_service.get_source_config()
        sink = config_service.get_sink_config()

        assert source.source_type == SourceType.DISCONNECTED
        assert sink.sink_type == SinkType.DISCONNECTED

    def test_set_source_success(self, fxt_config_service, fxt_source_config):
        """Test setting source configuration successfully."""
        old_source_id = fxt_config_service.get_source_config().id
        fxt_config_service.set_source_config(fxt_source_config)
        fxt_config_service._load_app_config()  # trigger DB loading explicitly

        source_config = fxt_config_service.get_source_config()
        assert old_source_id != source_config.id
        assert source_config.device_id == fxt_source_config.device_id

    def test_set_sink_success(self, fxt_config_service, fxt_sink_config):
        """Test setting sink configuration successfully."""
        old_sink_id = fxt_config_service.get_sink_config().id
        fxt_config_service.set_sink_config(fxt_sink_config)
        fxt_config_service._load_app_config()  # trigger DB loading explicitly

        sink_config = fxt_config_service.get_sink_config()
        assert old_sink_id != sink_config.id
        assert sink_config.rate_limit == fxt_sink_config.rate_limit
        assert sink_config.output_formats == fxt_sink_config.output_formats
        assert sink_config.broker_host == fxt_sink_config.broker_host
        assert sink_config.broker_port == fxt_sink_config.broker_port
        assert sink_config.topic == fxt_sink_config.topic
