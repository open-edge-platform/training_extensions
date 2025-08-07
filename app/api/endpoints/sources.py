"""Endpoints for managing pipeline sources"""

import logging
from http import HTTPStatus
from typing import Annotated
from uuid import UUID

import yaml
from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, Response

from app.api.dependencies import get_source_id
from app.schemas import Source, SourceType
from app.schemas.source import SourceAdapter
from app.services import ConfigurationService

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


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Source created", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
    },
)
async def create_source(
    source_config: Annotated[
        Source, Body(description=CREATE_SOURCE_BODY_DESCRIPTION, openapi_examples=CREATE_SOURCE_BODY_EXAMPLES)
    ],
) -> Source:
    """Create and configure a new source"""
    if source_config.source_type == SourceType.DISCONNECTED:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="The source with source_type=DISCONNECTED cannot be created"
        )

    return ConfigurationService.create_source(source_config)


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available source configurations", "model": list[Source]},
    },
)
async def list_sources() -> list[Source]:
    """List the available sources"""
    return ConfigurationService.list_sources()


@router.get(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source found", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def get_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> Source:
    """Get info about a source"""
    source = ConfigurationService.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Source with ID {source_id} not found")
    return source


@router.patch(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source successfully updated", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def update_source(
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
    if "source_type" in source_config:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="The 'source_type' field cannot be changed")
    source = ConfigurationService.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Source with ID {source_id} not found")
    return ConfigurationService.update_source(source, source_config)


@router.post(
    "/{source_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Source configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def export_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> Response:
    """Export a source to file"""
    source = ConfigurationService.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source with ID {source_id} not found")

    yaml_content = yaml.safe_dump(source.model_dump(mode="json"))

    return Response(
        content=yaml_content.encode("utf8"),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=source_{source_id}.yaml"},
    )


@router.post(":import")
async def import_source(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the source configuration")],
) -> Source:
    """Import a source from file"""
    try:
        yaml_content = await yaml_file.read()
        source_data = yaml.safe_load(yaml_content)

        source_config = SourceAdapter.validate_python(source_data)
        if source_config.source_type == SourceType.DISCONNECTED:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="The source with source_type=DISCONNECTED cannot be imported"
            )
        return ConfigurationService.create_source(source_config)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")


@router.delete(
    "/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_200_OK: {
            "description": "Source configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or source is used by at least one pipeline"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def delete_source(source_id: Annotated[UUID, Depends(get_source_id)]) -> None:
    """Remove a source"""
    source = ConfigurationService.get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Source with ID {source_id} not found")
    try:
        ConfigurationService.delete_source_by_id(source_id)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
