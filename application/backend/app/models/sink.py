# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from app.core.models import BaseRequiredIDNameModel
from app.models.base import BaseEntity


class OutputFormat(StrEnum):
    IMAGE_ORIGINAL = "image_original"
    IMAGE_WITH_PREDICTIONS = "image_with_predictions"
    PREDICTIONS = "predictions"


class SinkType(StrEnum):
    DISCONNECTED = "disconnected"
    FOLDER = "folder"
    MQTT = "mqtt"
    ROS = "ros"
    WEBHOOK = "webhook"


class BaseSinkConfig(BaseRequiredIDNameModel, BaseEntity):
    sink_type: SinkType
    rate_limit: float | None = None  # Rate limit in Hz, None means no limit
    output_formats: list[OutputFormat]


class SinkConfig(BaseModel):
    pass


class DisconnectedConfig(SinkConfig):
    pass


class DisconnectedSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.DISCONNECTED] = SinkType.DISCONNECTED
    id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    name: str = "No Sink"
    output_formats: list[OutputFormat] = []
    config_data: DisconnectedConfig = DisconnectedConfig()


class FolderConfig(SinkConfig):
    folder_path: str


class FolderSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.FOLDER]
    config_data: FolderConfig


class MqttConfig(SinkConfig):
    broker_host: str
    broker_port: int
    topic: str
    auth_required: bool = False


class MqttSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.MQTT]
    config_data: MqttConfig


class RosConfig(SinkConfig):
    topic: str


class RosSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.ROS]
    config_data: RosConfig


HttpMethod = Literal["POST", "PUT", "PATCH"]
HttpHeaders = dict[str, str]


class WebhookConfig(SinkConfig):
    webhook_url: str
    http_method: HttpMethod = "POST"
    headers: HttpHeaders | None = None
    timeout: int = 10  # seconds


class WebhookSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.WEBHOOK]
    config_data: WebhookConfig


Sink = Annotated[
    FolderSinkConfig | MqttSinkConfig | RosSinkConfig | WebhookSinkConfig | DisconnectedSinkConfig,
    Field(discriminator="sink_type"),
]

SinkAdapter: TypeAdapter[Sink] = TypeAdapter(Sink)
