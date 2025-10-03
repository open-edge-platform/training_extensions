# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.schemas import OutputFormat, SinkType
from app.schemas.sink import MQTT_PASSWORD, MQTT_USERNAME, MqttSinkConfig


@pytest.fixture
def fxt_mqtt_sink_config():
    return MqttSinkConfig(
        sink_type=SinkType.MQTT,
        id=uuid4(),
        name="Test MQTT Sink",
        broker_host="mqtt.example.com",
        broker_port=1883,
        topic="test/topic",
        output_formats=[OutputFormat.PREDICTIONS],
        auth_required=True,
    )


class TestMqttSinkConfig:
    """Test cases for MqttSinkConfig."""

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
    def test_get_invalid_credentials(self, env_vars, description, fxt_mqtt_sink_config):
        """Test error cases for invalid or missing credentials."""
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(RuntimeError, match="MQTT credentials not provided"),
        ):
            fxt_mqtt_sink_config.get_credentials()

    @patch.dict(os.environ, {MQTT_USERNAME: "testuser", MQTT_PASSWORD: "testpass"})
    def test_get_configured_stream_url_with_auth_success(self, fxt_mqtt_sink_config):
        """Test valid credentials."""
        assert fxt_mqtt_sink_config.get_credentials() == ("testuser", "testpass")
