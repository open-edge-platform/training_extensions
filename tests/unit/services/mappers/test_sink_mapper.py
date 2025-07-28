from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.db.schema import SinkDB
from app.schemas.configuration.output_config import FolderOutputConfig, MqttOutputConfig, SinkType
from app.services.mappers.sink_mapper import SinkMapper


class TestSinkMapper:
    """Test suite for SinkMapper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = SinkMapper()

    @pytest.mark.parametrize(
        "schema_instance,expected_model",
        [
            (
                FolderOutputConfig(
                    sink_type=SinkType.FOLDER,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    folder_path="/test/path",
                ),
                SinkDB(
                    sink_type=SinkType.FOLDER.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"folder_path": "/test/path"},
                ),
            ),
            (
                MqttOutputConfig(
                    sink_type=SinkType.MQTT,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    broker_host="localhost",
                    broker_port=1883,
                    topic="topic",
                ),
                SinkDB(
                    sink_type=SinkType.MQTT.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic"},
                ),
            ),
        ],
    )
    def test_from_schema_valid_sink_types(self, schema_instance, expected_model):
        """Test from_schema with valid sink types."""
        sink_id = str(uuid4())
        result = self.mapper.from_schema(schema_instance, sink_id=sink_id)

        assert isinstance(result, SinkDB)
        assert result.id == sink_id
        assert result.sink_type == expected_model.sink_type
        assert result.rate_limit == expected_model.rate_limit
        assert result.output_formats == expected_model.output_formats
        assert result.config_data == expected_model.config_data

    def test_from_schema_none_sink_raises_error(self):
        """Test from_schema raises ValueError when sink is None."""
        with pytest.raises(ValueError, match="Sink config cannot be None"):
            self.mapper.from_schema(None)

    def test_from_schema_unsupported_sink_type(self):
        """Test from_schema raises ValueError for unsupported sink type."""
        mock = MagicMock()
        mock.sink_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported sink type: UNSUPPORTED_TYPE"):
            self.mapper.from_schema(mock)

    @pytest.mark.parametrize(
        "db_instance,expected_schema",
        [
            (
                SinkDB(
                    sink_type=SinkType.FOLDER.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"folder_path": "/test/path"},
                ),
                FolderOutputConfig(
                    sink_type=SinkType.FOLDER,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    folder_path="/test/path",
                ),
            ),
            (
                SinkDB(
                    sink_type=SinkType.MQTT.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic"},
                ),
                MqttOutputConfig(
                    sink_type=SinkType.MQTT,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    broker_host="localhost",
                    broker_port=1883,
                    topic="topic",
                ),
            ),
        ],
    )
    def test_to_schema_valid_sink_types(self, db_instance, expected_schema):
        """Test to_schema with valid sink types."""
        result = self.mapper.to_schema(db_instance)

        assert result.sink_type == expected_schema.sink_type
        assert result.rate_limit == expected_schema.rate_limit
        assert result.output_formats == expected_schema.output_formats
        match result.sink_type:
            case SinkType.FOLDER:
                assert isinstance(result, FolderOutputConfig)
                assert result.folder_path == expected_schema.folder_path
            case SinkType.MQTT:
                assert isinstance(result, MqttOutputConfig)
                assert result.broker_host == expected_schema.broker_host
                assert result.broker_port == expected_schema.broker_port
                assert result.topic == expected_schema.topic

    def test_to_schema_unsupported_sink_type(self):
        """Test to_schema raises ValueError for unsupported sink type."""
        mock = MagicMock()
        mock.sink_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported sink type: UNSUPPORTED_TYPE"):
            self.mapper.to_schema(mock)
