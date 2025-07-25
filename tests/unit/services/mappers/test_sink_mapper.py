from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.db.schema import SinkDB
from app.schemas.configuration.output_config import DestinationType, FolderOutputConfig, MqttOutputConfig
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
                    destination_type=DestinationType.FOLDER,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    folder_path="/test/path",
                ),
                SinkDB(
                    destination_type=DestinationType.FOLDER.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"folder_path": "/test/path"},
                ),
            ),
            (
                MqttOutputConfig(
                    destination_type=DestinationType.MQTT,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    broker_host="localhost",
                    broker_port=1883,
                    topic="topic",
                ),
                SinkDB(
                    destination_type=DestinationType.MQTT.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic"},
                ),
            ),
        ],
    )
    def test_from_schema_valid_destination_types(self, schema_instance, expected_model):
        """Test from_schema with valid destination types."""
        sink_id = str(uuid4())
        result = self.mapper.from_schema(schema_instance, sink_id=sink_id)

        assert isinstance(result, SinkDB)
        assert result.id == sink_id
        assert result.destination_type == expected_model.destination_type
        assert result.rate_limit == expected_model.rate_limit
        assert result.output_formats == expected_model.output_formats
        assert result.config_data == expected_model.config_data

    def test_from_schema_none_sink_raises_error(self):
        """Test from_schema raises ValueError when sink is None."""
        with pytest.raises(ValueError, match="Sink config cannot be None"):
            self.mapper.from_schema(None)

    def test_from_schema_unsupported_destination_type(self):
        """Test from_schema raises ValueError for unsupported destination type."""
        mock = MagicMock()
        mock.destination_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported destination type: UNSUPPORTED_TYPE"):
            self.mapper.from_schema(mock)

    @pytest.mark.parametrize(
        "db_instance,expected_schema",
        [
            (
                SinkDB(
                    destination_type=DestinationType.FOLDER.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"folder_path": "/test/path"},
                ),
                FolderOutputConfig(
                    destination_type=DestinationType.FOLDER,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    folder_path="/test/path",
                ),
            ),
            (
                SinkDB(
                    destination_type=DestinationType.MQTT.value,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic"},
                ),
                MqttOutputConfig(
                    destination_type=DestinationType.MQTT,
                    rate_limit=0.2,
                    output_formats=["image_original", "image_with_predictions", "predictions"],
                    broker_host="localhost",
                    broker_port=1883,
                    topic="topic",
                ),
            ),
        ],
    )
    def test_to_schema_valid_destination_types(self, db_instance, expected_schema):
        """Test to_schema with valid destination types."""
        result = self.mapper.to_schema(db_instance)

        assert result.destination_type == expected_schema.destination_type
        assert result.rate_limit == expected_schema.rate_limit
        assert result.output_formats == expected_schema.output_formats
        match result.destination_type:
            case DestinationType.FOLDER:
                assert isinstance(result, FolderOutputConfig)
                assert result.folder_path == expected_schema.folder_path
            case DestinationType.MQTT:
                assert isinstance(result, MqttOutputConfig)
                assert result.broker_host == expected_schema.broker_host
                assert result.broker_port == expected_schema.broker_port
                assert result.topic == expected_schema.topic

    def test_to_schema_unsupported_destination_type(self):
        """Test to_schema raises ValueError for unsupported destination type."""
        mock = MagicMock()
        mock.destination_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported destination type: UNSUPPORTED_TYPE"):
            self.mapper.to_schema(mock)
