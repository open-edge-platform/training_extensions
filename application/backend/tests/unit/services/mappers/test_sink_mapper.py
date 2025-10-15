# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.db.schema import SinkDB
from app.schemas.sink import FolderSinkConfig, MqttSinkConfig, OutputFormat, SinkType, WebhookSinkConfig
from app.services.mappers.sink_mapper import SinkMapper

SINK_ID = uuid4()
SUPPORTED_SINKS_MAPPING = [
    (
        FolderSinkConfig(
            sink_type=SinkType.FOLDER,
            id=SINK_ID,
            name="Test Folder Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            folder_path="/test/path",
        ),
        SinkDB(
            sink_type=SinkType.FOLDER,
            id=str(SINK_ID),
            name="Test Folder Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={"folder_path": "/test/path"},
        ),
    ),
    (
        MqttSinkConfig(
            sink_type=SinkType.MQTT,
            id=SINK_ID,
            name="Test Mqtt Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            broker_host="localhost",
            broker_port=1883,
            topic="topic",
            auth_required=False,
        ),
        SinkDB(
            sink_type=SinkType.MQTT,
            id=str(SINK_ID),
            name="Test Mqtt Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic", "auth_required": False},
        ),
    ),
    (
        WebhookSinkConfig(
            sink_type=SinkType.WEBHOOK,
            id=SINK_ID,
            name="Test Webhook Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            webhook_url="https://example.com/webhook",
            http_method="PATCH",
            headers={"Authorization": "Bearer token"},
            timeout=5,
        ),
        SinkDB(
            sink_type=SinkType.WEBHOOK,
            id=str(SINK_ID),
            name="Test Webhook Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={
                "webhook_url": "https://example.com/webhook",
                "http_method": "PATCH",
                "headers": {"Authorization": "Bearer token"},
                "timeout": 5,
            },
        ),
    ),
]


class TestSinkMapper:
    """Test suite for SinkMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_model", SUPPORTED_SINKS_MAPPING.copy())
    def test_from_schema_valid_sink_types(self, schema_instance, expected_model):
        """Test from_schema with valid sink types."""
        result = SinkMapper.from_schema(schema_instance)

        assert isinstance(result, SinkDB)
        assert result.id == expected_model.id
        assert result.name == expected_model.name
        assert result.sink_type == expected_model.sink_type
        assert result.rate_limit == expected_model.rate_limit
        assert result.output_formats == expected_model.output_formats
        assert result.config_data == expected_model.config_data

    def test_from_schema_none_sink_raises_error(self):
        """Test from_schema raises ValueError when sink is None."""
        with pytest.raises(ValueError, match="Sink config cannot be None"):
            SinkMapper.from_schema(None)

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_SINKS_MAPPING.copy()])
    def test_to_schema_valid_sink_types(self, db_instance, expected_schema):
        """Test to_schema with valid sink types."""
        result = SinkMapper.to_schema(db_instance)

        assert result.id == expected_schema.id
        assert result.name == expected_schema.name
        assert result.sink_type == expected_schema.sink_type
        assert result.rate_limit == expected_schema.rate_limit
        assert result.output_formats == expected_schema.output_formats
        match result.sink_type:
            case SinkType.FOLDER:
                assert isinstance(result, FolderSinkConfig)
                assert result.folder_path == expected_schema.folder_path
            case SinkType.MQTT:
                assert isinstance(result, MqttSinkConfig)
                assert result.broker_host == expected_schema.broker_host
                assert result.broker_port == expected_schema.broker_port
                assert result.topic == expected_schema.topic
            case SinkType.WEBHOOK:
                assert isinstance(result, WebhookSinkConfig)
                assert result.webhook_url == expected_schema.webhook_url
                assert result.http_method == expected_schema.http_method
                assert result.headers == expected_schema.headers
                assert result.timeout == expected_schema.timeout
