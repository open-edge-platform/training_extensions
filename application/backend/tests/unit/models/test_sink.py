# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models.sink import MQTT_PASSWORD, MQTT_USERNAME, MqttConfig, MqttSinkConfig, OutputFormat, SinkType


@pytest.fixture
def fxt_mqtt_sink_config():
    return MqttSinkConfig(
        sink_type=SinkType.MQTT,
        id=uuid4(),
        name="Test MQTT Sink",
        output_formats=[OutputFormat.PREDICTIONS],
        config_data=MqttConfig(
            broker_host="mqtt.example.com",
            broker_port=1883,
            topic="test/topic",
            auth_required=True,
        ),
    )


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
def test_mqtt_config_get_invalid_credentials(env_vars, description, fxt_mqtt_sink_config):
    """Test error cases for invalid or missing credentials."""
    with (
        patch.dict(os.environ, env_vars, clear=True),
        pytest.raises(RuntimeError, match="MQTT credentials not provided"),
    ):
        fxt_mqtt_sink_config.config_data.get_credentials()


@patch.dict(os.environ, {MQTT_USERNAME: "testuser", MQTT_PASSWORD: "testpass"})
def test_get_configured_stream_url_with_auth_success(fxt_mqtt_sink_config):
    """Test valid credentials."""
    assert fxt_mqtt_sink_config.config_data.get_credentials() == ("testuser", "testpass")
