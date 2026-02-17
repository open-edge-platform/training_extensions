# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from pydantic import BaseModel, computed_field

from app.core.models import BaseRequiredIDNameModel, Pagination
from app.models import DatasetItemAnnotation, MediaFormat, MediaType


class MediaView(BaseRequiredIDNameModel):
    """
    Media
    """

    type: MediaType
    format: MediaFormat
    width: int
    height: int
    size: int
    fps: float | None
    frame_count: int | None
    source_id: UUID | None = None

    @computed_field
    @property
    def duration(self) -> float | None:
        """Return duration in seconds"""
        return self.frame_count / self.fps if self.frame_count is not None and self.fps is not None else None

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
