# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from uuid import UUID

from .base import BaseEntity


class ImageFormat(StrEnum):
    """Format of the image."""

    JPG = "jpg"
    PNG = "png"


MediaFormat = ImageFormat


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
    source_id: UUID | None
