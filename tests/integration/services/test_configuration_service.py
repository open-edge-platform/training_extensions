import multiprocessing as mp
from unittest.mock import patch

import pytest

from app.schemas.configuration import OutputFormat
from app.schemas.configuration.input_config import SourceType, WebcamSourceConfig
from app.schemas.configuration.output_config import MqttOutputConfig, SinkType
from app.services.configuration_service import ConfigurationService


@pytest.fixture(scope="function", autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.configuration_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock


@pytest.fixture
def sample_source_config() -> WebcamSourceConfig:
    """Sample source configuration data."""
    return WebcamSourceConfig(source_type=SourceType.WEBCAM, device_id=1)


@pytest.fixture
def sample_sink_config() -> MqttOutputConfig:
    """Sample sink configuration data."""
    return MqttOutputConfig(
        sink_type=SinkType.MQTT,
        rate_limit=0.1,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        broker_host="localhost",
        broker_port=1883,
        topic="topic",
    )


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_load_app_config_success(self):
        """Test successful loading of application configuration after migration applied."""
        config_service = ConfigurationService()

        app_config = config_service.get_app_config()

        assert app_config is not None
        assert app_config.input is not None
        assert app_config.output is not None
        assert app_config.input.source_type == SourceType.VIDEO_FILE
        assert app_config.output.sink_type == SinkType.FOLDER

    def test_set_source_success(self, sample_source_config):
        """Test setting source configuration successfully."""
        config_service = ConfigurationService()
        config_service.config_changed_condition = mp.Condition()

        old_source_id = config_service._source_id
        config_service.set_source_config(sample_source_config)
        config_service._load_app_config()  # trigger DB loading explicitly
        assert old_source_id != config_service._source_id

        source_config = config_service.get_source_config()
        assert source_config.device_id == sample_source_config.device_id

    def test_set_sink_success(self, sample_sink_config):
        """Test setting sink configuration successfully."""
        config_service = ConfigurationService()
        config_service.config_changed_condition = mp.Condition()

        old_sink_id = config_service._sink_id
        config_service.set_sink_config(sample_sink_config)
        config_service._load_app_config()  # trigger DB loading explicitly
        assert old_sink_id != config_service._sink_id

        sink_config = config_service.get_sink_config()
        assert sink_config.rate_limit == sample_sink_config.rate_limit
        assert sink_config.output_formats == sample_sink_config.output_formats
        assert sink_config.broker_host == sample_sink_config.broker_host
        assert sink_config.broker_port == sample_sink_config.broker_port
        assert sink_config.topic == sample_sink_config.topic
