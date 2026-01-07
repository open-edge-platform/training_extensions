# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from os import getenv
from typing import Annotated, Literal
from urllib.parse import urlparse, urlunparse
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter

from app.core.models import BaseRequiredIDNameModel

from .base import BaseEntity

IP_CAMERA_USERNAME = "IP_CAMERA_USERNAME"
IP_CAMERA_PASSWORD = "IP_CAMERA_PASSWORD"  # noqa: S105


class SourceType(StrEnum):
    DISCONNECTED = "disconnected"
    USB_CAMERA = "usb_camera"
    IP_CAMERA = "ip_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class BaseSourceConfig(BaseRequiredIDNameModel, BaseEntity):
    source_type: SourceType


class SourceConfig(BaseModel):
    pass


class DisconnectedConfig(SourceConfig):
    pass


class DisconnectedSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.DISCONNECTED] = SourceType.DISCONNECTED
    id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    name: str = "No Source"
    config_data: DisconnectedConfig = DisconnectedConfig()


class USBCameraConfig(SourceConfig):
    device_id: int
    codec: Annotated[str | None, StringConstraints(min_length=4, max_length=4, to_upper=True)] = Field(
        None, description="Video codec fourcc"
    )


class USBCameraSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.USB_CAMERA]
    config_data: USBCameraConfig


class IPCameraConfig(SourceConfig):
    stream_url: str
    auth_required: bool = False

    def get_configured_stream_url(self) -> str:
        """Configure stream URL with authentication if required."""
        if not self.auth_required:
            return self.stream_url

        username = getenv(IP_CAMERA_USERNAME)
        password = getenv(IP_CAMERA_PASSWORD)

        if not username or not password:
            raise RuntimeError("IP camera credentials not provided.")

        # Modify the stream URL to include authentication
        uri = urlparse(self.stream_url)
        netloc = f"{username}:{password}@{uri.netloc}"
        return urlunparse((uri.scheme, netloc, uri.path, uri.params, uri.query, uri.fragment))


class IPCameraSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.IP_CAMERA]
    config_data: IPCameraConfig


class VideoFileConfig(SourceConfig):
    video_path: str


class VideoFileSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.VIDEO_FILE]
    config_data: VideoFileConfig


class ImagesFolderConfig(SourceConfig):
    images_folder_path: str
    ignore_existing_images: bool


class ImagesFolderSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    config_data: ImagesFolderConfig


Source = Annotated[
    USBCameraSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig
    | DisconnectedSourceConfig,
    Field(discriminator="source_type"),
]

SourceAdapter: TypeAdapter[Source] = TypeAdapter(Source)
