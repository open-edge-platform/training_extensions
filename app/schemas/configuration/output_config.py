from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DestinationType(str, Enum):
    DISCONNECTED = "disconnected"
    FOLDER = "folder"
    MQTT = "mqtt"
    DDS = "dds"
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
    destination_type: Literal[DestinationType.DISCONNECTED] = DestinationType.DISCONNECTED


class FolderOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.FOLDER]
    folder_path: str


class MqttOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.MQTT]
    broker_host: str
    broker_port: int
    topic: str
    username: str | None = None
    password: str | None = None


class DdsOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.DDS]
    dds_topic: str


class RosOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.ROS]
    ros_topic: str


class WebhookOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.WEBHOOK]
    webhook_url: str


Sink = Annotated[
    DisconnectedOutputConfig
    | FolderOutputConfig
    | MqttOutputConfig
    | DdsOutputConfig
    | RosOutputConfig
    | WebhookOutputConfig,
    Field(discriminator="destination_type"),
]
