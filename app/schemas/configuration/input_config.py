from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


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


class IpCameraSourceConfig(BaseModel):
    source_type: Literal[SourceType.IP_CAMERA]
    device_url: str


class VideoFileSourceConfig(BaseModel):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str


class ImagesFolderSourceConfig(BaseModel):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str


InputConfig = Annotated[
    DisconnectedSourceConfig
    | WebcamSourceConfig
    | IpCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig,
    Field(discriminator="source_type"),
]
