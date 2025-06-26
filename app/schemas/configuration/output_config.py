from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DestinationType(str, Enum):
    DISCONNECTED = "disconnected"
    FOLDER = "folder"
    MQTT = "mqtt"
    DDS = "dds"
    ROS = "ros"


class OutputFormat(str, Enum):
    IMAGE_ORIGINAL = "image_original"
    IMAGE_WITH_PREDICTIONS = "image_with_predictions"
    PREDICTIONS = "predictions"


class BaseOutputConfig(BaseModel):
    output_formats: list[OutputFormat]
    rate_limit: float | None = None  # Rate limit in Hz, None means no limit


class FolderOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.FOLDER]
    folder_path: str


class MqttOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.MQTT]
    mqtt_broker_url: str
    mqtt_topic: str


class DdsOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.DDS]
    dds_topic: str


class RosOutputConfig(BaseOutputConfig):
    destination_type: Literal[DestinationType.ROS]
    ros_topic: str


OutputConfig = Annotated[
    FolderOutputConfig | MqttOutputConfig | DdsOutputConfig | RosOutputConfig, Field(discriminator="destination_type")
]
