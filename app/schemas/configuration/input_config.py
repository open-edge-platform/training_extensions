from enum import Enum
from os import getenv
from typing import Annotated, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field

IP_CAMERA_USERNAME = "IP_CAMERA_USERNAME"
IP_CAMERA_PASSWORD = "IP_CAMERA_PASSWORD"  # noqa: S105


class SourceType(str, Enum):
    DISCONNECTED = "disconnected"
    WEBCAM = "webcam"
    IP_CAMERA = "ip_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class DisconnectedSourceConfig(BaseModel):
    source_type: Literal[SourceType.DISCONNECTED]


class WebcamSourceConfig(BaseModel):
    source_type: Literal[SourceType.WEBCAM]
    device_id: int


class IPCameraSourceConfig(BaseModel):
    source_type: Literal[SourceType.IP_CAMERA]
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


class VideoFileSourceConfig(BaseModel):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str


class ImagesFolderSourceConfig(BaseModel):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str
    ignore_existing_images: bool


InputConfig = Annotated[
    DisconnectedSourceConfig
    | WebcamSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig,
    Field(discriminator="source_type"),
]
