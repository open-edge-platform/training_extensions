from typing import Any

from app.db.schema import SinkDB
from app.schemas.configuration.output_config import FolderOutputConfig, MqttOutputConfig, Sink, SinkType


class SinkMapper:
    """Mapper for Sink model <-> Sink schema conversions."""

    def to_schema(self, sink_db: SinkDB) -> Sink:
        """Convert Sink model to Sink schema."""

        match sink_db.sink_type:
            case SinkType.FOLDER.value:
                return FolderOutputConfig(
                    sink_type=SinkType.FOLDER,
                    output_formats=sink_db.output_formats,
                    rate_limit=sink_db.rate_limit,
                    folder_path=sink_db.config_data.get("folder_path", ""),
                )
            case SinkType.MQTT.value:
                return MqttOutputConfig(
                    sink_type=SinkType.MQTT,
                    output_formats=sink_db.output_formats,
                    rate_limit=sink_db.rate_limit,
                    broker_host=sink_db.config_data.get("broker_host", ""),
                    broker_port=int(sink_db.config_data.get("broker_port", 1883)),
                    topic=sink_db.config_data.get("topic", ""),
                )
            case _:
                raise ValueError(f"Unsupported sink type: {sink_db.sink_type}")

    def from_schema(self, sink: Sink, sink_id: str | None = None) -> SinkDB:
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
            id=sink_id,
            sink_type=sink.sink_type.value,
            output_formats=sink.output_formats,
            rate_limit=sink.rate_limit,
            config_data=config_data,
        )
