# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from app.core.models import BaseRequiredIDNameModel, Pagination
from app.models import DatasetItemAnnotation, MediaFormat, MediaType


class ImageView(BaseRequiredIDNameModel):
    """
    Image
    """

    type: Literal[MediaType.IMAGE]
    format: MediaFormat
    width: int
    height: int
    size: int
    source_id: UUID | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "img-010203",
                "type": "image",
                "format": "jpg",
                "width": 1280,
                "height": 720,
                "size": 2211840,
                "source_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
            }
        }
    }


class VideoFrameView(BaseRequiredIDNameModel):
    """
    VideoFrame
    """

    type: Literal[MediaType.VIDEO_FRAME]
    format: MediaFormat
    width: int
    height: int
    size: int
    source_id: UUID | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "test_video_frame_4",
                "type": "video_frame",
                "format": "jpg",
                "width": 1280,
                "height": 720,
                "size": 2211840,
                "source_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
            }
        }
    }


class VideoView(BaseRequiredIDNameModel):
    """
    Video
    """

    type: Literal[MediaType.VIDEO]
    format: MediaFormat
    width: int
    height: int
    size: int
    fps: float
    frame_count: int
    source_id: UUID | None = None
    duration: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fd",
                "name": "test_video",
                "type": "video",
                "format": "avi",
                "width": 1280,
                "height": 720,
                "fps": 25.0,
                "frame_count": 100,
                "duration": 4.0,
                "size": 2211840,
                "source_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
            }
        }
    }


MediaView = Annotated[ImageView | VideoView | VideoFrameView, Field(discriminator="type")]

MediaViewAdapter: TypeAdapter[MediaView] = TypeAdapter(MediaView)


class MediaWithPagination(BaseModel):
    """
    Media list with pagination info
    """

    items: list[MediaView]
    pagination: Pagination


class SetMediaAnnotations(BaseModel):
    """Schema for setting media annotations"""

    annotations: list[DatasetItemAnnotation]

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotations": [
                    {
                        "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                        "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                    }
                ]
            }
        }
    }


class MediaAnnotations(BaseModel):
    """
    Media annotations with information about source
    """

    annotations: list[DatasetItemAnnotation]
    user_reviewed: bool
    prediction_model_id: UUID | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotations": [
                    {
                        "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                        "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                    }
                ],
                "user_reviewed": False,
                "prediction_model_id": None,
            }
        }
    }
