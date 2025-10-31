# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.models import MqttSinkConfig, OutputFormat, SinkType
from app.models.sink import MqttConfig


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


class TestMqttSinkConfig:
    """Test cases for MqttSinkConfig."""
