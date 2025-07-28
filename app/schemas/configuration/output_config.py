from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SinkType(str, Enum):
    DISCONNECTED = "disconnected"
    FOLDER = "folder"
    MQTT = "mqtt"
    ROS = "ros"
    WEBHOOK = "webhook"


class OutputFormat(str, Enum):
    IMAGE_ORIGINAL = "image_original"
    IMAGE_WITH_PREDICTIONS = "image_with_predictions"
    PREDICTIONS = "predictions"


class BaseOutputConfig(BaseModel):
    output_formats: list[OutputFormat]
    rate_limit: float | None = None  # Rate limit in Hz, None means no limit


class DisconnectedOutputConfig(BaseOutputConfig):
    sink_type: Literal[SinkType.DISCONNECTED] = SinkType.DISCONNECTED


class FolderOutputConfig(BaseOutputConfig):
    sink_type: Literal[SinkType.FOLDER]
    folder_path: str


class MqttOutputConfig(BaseOutputConfig):
    sink_type: Literal[SinkType.MQTT]
    broker_host: str
    broker_port: int
    topic: str
    username: str | None = None
    password: str | None = None


class RosOutputConfig(BaseOutputConfig):
    sink_type: Literal[SinkType.ROS]
    ros_topic: str


class WebhookOutputConfig(BaseOutputConfig):
    sink_type: Literal[SinkType.WEBHOOK]
    webhook_url: str


Sink = Annotated[
    DisconnectedOutputConfig | FolderOutputConfig | MqttOutputConfig | RosOutputConfig | WebhookOutputConfig,
    Field(discriminator="sink_type"),
]
