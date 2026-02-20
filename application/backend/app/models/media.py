# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from uuid import UUID

from pydantic import computed_field

from .base import BaseEntity


class ImageFormat(StrEnum):
    """Format of the image."""

    JPG = "jpg"
    PNG = "png"


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


class Media(BaseEntity):
    """
    Media represents an uploaded or fetched media within a dataset.

    Attributes:
        id: Unique identifier for the media.
        type: Type of the media.
        project_id: Identifier of the project to which the media belongs.
        name: Name of the media.
        format: Format of the media (e.g., JPG, PNG).
        width: Width of the media in pixels.
        height: Height of the media in pixels.
        size: Size of the media in bytes.
        fps: Video fps (applicable only for video media).
        frame_count: Total number of frames (applicable only for video media).
        source_id: Identifier of the source from which the media was acquired, if applicable.
    """

    id: UUID
    type: MediaType
    project_id: UUID
    name: str
    format: MediaFormat
    width: int
    height: int
    size: int
    fps: float | None
    frame_count: int | None
    source_id: UUID | None

    @computed_field
    @property
    def duration(self) -> float | None:
        """Return duration in seconds"""
        return self.frame_count / self.fps if self.frame_count is not None and self.fps is not None else None


class VideoFrame(BaseEntity):
    id: UUID
    video_id: UUID
    frame_index: int
