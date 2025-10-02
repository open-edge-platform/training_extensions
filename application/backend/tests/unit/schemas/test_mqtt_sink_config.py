import os
from unittest.mock import patch

import pytest

from app.schemas import OutputFormat, SinkType
from app.schemas.sink import MQTT_PASSWORD, MQTT_USERNAME, MqttSinkConfig


class TestMqttSinkConfig:
    """Test cases for MqttSinkConfig."""

    def test_basic_initialization(self):
        """Test basic config initialization without auth."""
        config = MqttSinkConfig(
            sink_type=SinkType.MQTT,
            broker_host="mqtt.example.com",
            broker_port=1883,
            topic="test/topic",
            output_formats=[OutputFormat.PREDICTIONS],
        )

        assert config.sink_type == SinkType.MQTT
        assert config.broker_host == "mqtt.example.com"
        assert config.broker_port == 1883
        assert config.topic == "test/topic"
        assert config.output_formats == [OutputFormat.PREDICTIONS]
        assert not config.auth_required

    def test_default_auth_required_value(self):
        """Test that auth_required defaults to False."""
        config = MqttSinkConfig(
            sink_type=SinkType.MQTT,
            broker_host="mqtt.example.com",
            broker_port=1883,
            topic="test/topic",
            output_formats=[OutputFormat.PREDICTIONS],
            auth_required=False,
        )

        assert not config.auth_required

    @pytest.mark.parametrize(
        "env_vars,description",
        [
            ({}, "both username and password missing"),
            ({MQTT_USERNAME: "testuser"}, "password missing"),
            ({MQTT_PASSWORD: "testpass"}, "username missing"),
            ({MQTT_USERNAME: "", MQTT_PASSWORD: "testpass"}, "username is empty"),
            ({MQTT_USERNAME: "testuser", MQTT_PASSWORD: ""}, "password is empty"),
        ],
    )
    def test_get_invalid_credentials(self, env_vars, description):
        """Test error cases for invalid or missing credentials."""
        with patch.dict(os.environ, env_vars, clear=True):
            config = MqttSinkConfig(
                sink_type=SinkType.MQTT,
                broker_host="mqtt.example.com",
                broker_port=1883,
                topic="test/topic",
                output_formats=[OutputFormat.PREDICTIONS],
                auth_required=True,
            )

            with pytest.raises(RuntimeError, match="MQTT credentials not provided"):
                config.get_credentials()

    @patch.dict(os.environ, {MQTT_USERNAME: "testuser", MQTT_PASSWORD: "testpass"})
    def test_get_configured_stream_url_with_auth_success(self):
        """Test valid credentials."""
        config = MqttSinkConfig(
            sink_type=SinkType.MQTT,
            broker_host="mqtt.example.com",
            broker_port=1883,
            topic="test/topic",
            output_formats=[OutputFormat.PREDICTIONS],
            auth_required=True,
        )

        assert config.get_credentials() == ("testuser", "testpass")
