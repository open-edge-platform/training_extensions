from typing import Any
from uuid import UUID

from app.db.schema import SinkDB
from app.schemas.sink import FolderSinkConfig, MqttSinkConfig, Sink, SinkType


class SinkMapper:
    """Mapper for Sink model <-> Sink schema conversions."""

    @staticmethod
    def to_schema(sink_db: SinkDB) -> Sink:
        """Convert Sink model to Sink schema."""

        match sink_db.sink_type:
            case SinkType.FOLDER.value:
                return FolderSinkConfig(
                    id=UUID(sink_db.id),
                    sink_type=SinkType.FOLDER,
                    output_formats=sink_db.output_formats,
                    rate_limit=sink_db.rate_limit,
                    folder_path=sink_db.config_data.get("folder_path", ""),
                )
            case SinkType.MQTT.value:
                return MqttSinkConfig(
                    id=UUID(sink_db.id),
                    sink_type=SinkType.MQTT,
                    output_formats=sink_db.output_formats,
                    rate_limit=sink_db.rate_limit,
                    broker_host=sink_db.config_data.get("broker_host", ""),
                    broker_port=int(sink_db.config_data.get("broker_port", 1883)),
                    topic=sink_db.config_data.get("topic", ""),
                )
            case _:
                raise ValueError(f"Unsupported sink type: {sink_db.sink_type}")

    @staticmethod
    def from_schema(sink: Sink) -> SinkDB:
        """Convert Sink schema to Sink model."""
        if sink is None:
            raise ValueError("Sink config cannot be None")

        config_data: dict[str, Any] = {}

        match sink.sink_type:
            case SinkType.FOLDER:
                config_data["folder_path"] = sink.folder_path
            case SinkType.MQTT:
                config_data.update(
                    {
                        "broker_host": sink.broker_host,
                        "broker_port": sink.broker_port,
                        "topic": sink.topic,
                    }
                )
            case _:
                raise ValueError(f"Unsupported sink type: {sink.sink_type}")

        return SinkDB(
            id=str(sink.id),
            sink_type=sink.sink_type.value,
            name=sink.name,
            output_formats=sink.output_formats,
            rate_limit=sink.rate_limit,
            config_data=config_data,
        )
