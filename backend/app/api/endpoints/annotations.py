# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""API Endpoints for Media Annotations"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from fastapi.openapi.models import Example

from app.api.dependencies import get_media_id
from app.api.tags import Tags
from app.schemas.base import BaseIDModel

router = APIRouter(prefix="/api/media/{media_id}/annotations", tags=[Tags.ANNOTATIONS])

CREATE_ANNOTATION_BODY_DESCRIPTION = """
Configuration for the new annotation. The exact fields depend on the type of annotation being created.
- For `full_image` annotations, only the `label` field is required.
- For `rectangle` annotations, the `label`, `x`, `y`, `width`, and `height` fields are required.
- For `polygon` annotations, the `label` field is required, along with a list of `points` defining the polygon vertices.
"""
CREATE_ANNOTATION_BODY_EXAMPLES = {
    "classification": Example(
        summary="Full image annotation",
        description="Configuration for a complete annotation of an image with label",
        value={
            "label": "Car",
            "type": "full_image",
        },
    ),
    "detection": Example(
        summary="Object detection annotation",
        description="Configuration for an object detection annotation with bounding boxes",
        value={
            "label": "Pedestrian",
            "type": "rectangle",
            "x": 100,
            "y": 150,
            "width": 200,
            "height": 100,
        },
    ),
    "segmentation": Example(
        summary="Image segmentation annotation",
        description="Configuration for an image segmentation annotation with polygon points",
        value={
            "label": "Car",
            "type": "polygon",
            "points": [
                [50, 50],
                [150, 50],
                [150, 150],
                [50, 150],
            ],
        },
    ),
}


# TODO: Replace the model with a proper one from `app.schemas` package when defined
class Annotation(BaseIDModel):
    """Represents an annotation for a media item."""


@router.get(
    "",
    response_model=list[Annotation],
    responses={
        status.HTTP_200_OK: {"description": "List of annotations for the media"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def list_annotations(media_id: Annotated[UUID, Depends(get_media_id)]) -> list[Annotation]:
    """List all annotations for a given media item."""
    _ = media_id
    raise NotImplementedError


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=Annotation,
    responses={
        status.HTTP_201_CREATED: {"description": "Annotation created successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def create_annotation(
    media_id: Annotated[UUID, Depends(get_media_id)],
    annotation: Annotated[
        Annotation,
        Body(description=CREATE_ANNOTATION_BODY_DESCRIPTION, openapi_examples=CREATE_ANNOTATION_BODY_EXAMPLES),
    ],
) -> Annotation:
    """Create a new annotation for a media item."""
    _ = annotation, media_id
    raise NotImplementedError


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Annotations deleted successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def delete_annotations(media_id: Annotated[UUID, Depends(get_media_id)]) -> None:
    """Delete all annotations for a media item."""
    _ = media_id
    raise NotImplementedError
