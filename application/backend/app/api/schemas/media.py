# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from pydantic import BaseModel

from app.core.models import BaseRequiredIDNameModel, Pagination
from app.models import MediaFormat, MediaType


class MediaView(BaseRequiredIDNameModel):
    """
    Media
    """

    type: MediaType
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


class MediaWithPagination(BaseModel):
    """
    Media list with pagination info
    """

    items: list[MediaView]
    pagination: Pagination
