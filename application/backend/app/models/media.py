# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field

from .base import BaseEntity


class ImageFormat(StrEnum):
    """Format of the image."""

    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    JFIF = "jfif"
    TIF = "tif"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"


class VideoFormat(StrEnum):
    """Format of the video."""

    MP4 = "mp4"
    AVI = "avi"
    MKV = "mkv"
    MOV = "mov"
    WEBM = "webm"
    M4V = "m4v"


MediaFormat = ImageFormat | VideoFormat


class MediaType(StrEnum):
    """Type of the media"""

    IMAGE = "image"
    VIDEO = "video"
    VIDEO_FRAME = "video_frame"


class BaseMedia(BaseEntity):
    """
    Media represents an uploaded or fetched media within a dataset, it can be one of Image, Video or VideoFrame.

    Attributes:
        id: Unique identifier for the media.
        type: Type of the media.
        project_id: Identifier of the project to which the media belongs.
        name: Name of the media.
        format: Format of the media (e.g., JPG, PNG).
        width: Width of the media in pixels.
        height: Height of the media in pixels.
        size: Size of the media in bytes.
        source_id: Identifier of the source from which the media was acquired, if applicable.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID
    project_id: UUID
    name: str
    format: MediaFormat
    width: int
    height: int
    size: int
    source_id: UUID | None


class Image(BaseMedia):
    type: Literal[MediaType.IMAGE]


class VideoFrame(BaseMedia):
    """
    Attributes:
        video_id: Video identifier
        frame_index: Frame index
    """

    type: Literal[MediaType.VIDEO_FRAME]
    video_id: UUID
    frame_index: int


class Video(BaseMedia):
    """
    Attributes:
        fps: Video fps
        frame_count: Total number of frames
    """

    type: Literal[MediaType.VIDEO]
    fps: float
    frame_count: int

    @computed_field
    @property
    def duration(self) -> float:
        """Return duration in seconds"""
        return self.frame_count / self.fps


class NotAnnotatedVideoFrame(BaseModel):
    model_config = ConfigDict(frozen=True)
    video: Video
    frame_index: int

    @computed_field
    @property
    def video_id(self) -> UUID:
        return self.video.id


Media = Annotated[Image | Video | VideoFrame, Field(discriminator="type")]

MediaAdapter: TypeAdapter[Media] = TypeAdapter(Media)
