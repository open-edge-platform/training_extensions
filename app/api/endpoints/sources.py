"""Endpoints for managing pipeline sources"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import get_source_id
from app.schemas.configuration import Source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources", tags=["Sources"])

CREATE_SOURCE_BODY_DESCRIPTION = """
Configuration for the new source. The exact list of fields that can be configured depends on the source type.
"""
CREATE_SOURCE_BODY_EXAMPLES = {
    "webcam": {
        "summary": "Webcam source",
        "description": "Configuration for a webcam source",
        "value": {
            "source_type": "webcam",
            "name": "My Webcam",
            "device_id": 0,
        },
    },
    "ip_camera": {
        "summary": "IP camera source",
        "description": "Configuration for an IP camera source",
        "value": {
            "source_type": "ip_camera",
            "name": "IP Camera 1",
            "stream_url": "rtsp://192.168.1.100:554/stream1",
            "auth_required": True,
        },
    },
    "video_file": {
        "summary": "Video file source",
        "description": "Configuration for a video file source",
        "value": {
            "source_type": "video_file",
            "name": "Camera recording 123",
            "video_path": "/path/to/video.mp4",
        },
    },
    "images_folder": {
        "summary": "Images folder source",
        "description": "Configuration for a folder containing images source",
        "value": {
            "source_type": "images_folder",
            "name": "Production Samples",
            "folder_path": "/path/to/images",
            "ignore_existing_images": True,
        },
    },
}

UPDATE_SOURCE_BODY_DESCRIPTION = """
Partial source configuration update. May contain any subset of fields from the respective source type
(e.g., 'device_id' for webcams; 'video_path' for video files).
Fields not included in the request will remain unchanged. The 'source_type' field cannot be changed.
"""
UPDATE_SOURCE_BODY_EXAMPLES = {
    "webcam": {
        "summary": "Update webcam source",
        "description": "Rename a webcam source",
        "value": {
            "name": "Updated Webcam Name",
        },
    },
    "video_file": {
        "summary": "Update video file source",
        "description": "Change the video path for a video file source",
        "value": {
            "video_path": "/new/path/to/video.mp4",
        },
    },
}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_source(
    source_config: Annotated[
        Source, Body(description=CREATE_SOURCE_BODY_DESCRIPTION, openapi_examples=CREATE_SOURCE_BODY_EXAMPLES)
    ],
) -> Source:
    """Create and configure a new source"""
    raise NotImplementedError


@router.get("")
def list_sources() -> list[Source]:
    """List the available sources"""
    raise NotImplementedError


@router.get("/{source_id}")
def get_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> Source:
    """Get info about a source"""
    raise NotImplementedError


@router.patch("/{source_id}")
def update_source(
    source_id: Annotated[UUID, Depends(get_source_id)],
    source_config: Annotated[
        dict,
        Body(
            description=(
                "Partial source configuration update. "
                "May contain any subset of fields from the respective source type "
                "(e.g., 'device_id' for webcams; 'video_path' for video files). "
                "Fields not included in the request will remain unchanged. "
                "The 'source_type' field cannot be changed."
            ),
            openapi_examples=UPDATE_SOURCE_BODY_EXAMPLES,
        ),
    ],
) -> Source:
    """Reconfigure an existing source"""
    raise NotImplementedError


@router.post(
    "/{source_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Source configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
def export_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> FileResponse:
    """Export a source to file"""
    raise NotImplementedError


@router.post(":import")
def import_source(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the source configuration")],
) -> Source:
    """Import a source from file"""
    raise NotImplementedError


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> None:
    """Remove a source"""
    raise NotImplementedError
