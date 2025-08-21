from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.dependencies import is_valid_uuid
from app.schemas.base import BaseIDNameModel

router = APIRouter(prefix="/api/media", tags=["Media"])

MEDIA_RESPONSE_LIST_EXAMPLES = {
    "basic": Example(
        summary="Paginated list of media items",
        description="A sample paginated response containing media items with metadata",
        value={
            "data": [
                {
                    "id": "b0feaabc-da2b-442e-9b3e-55c11c2c2ff2",
                    "name": "cat.jpg",
                },
                {
                    "id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
                    "name": "dog.png",
                },
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_items": 2,
                "total_pages": 1,
            },
        },
    ),
    "empty_response": Example(
        summary="Empty result with pagination",
        description="Response when no media items match the filter criteria",
        value={
            "data": [],
            "pagination": {
                "total_count": 0,
                "page": 1,
                "page_size": 20,
                "total_pages": 0,
            },
        },
    ),
}


def get_media_id(media_id: str) -> UUID:
    """Initializes and validates a media ID"""
    if not is_valid_uuid(media_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid media ID")
    return UUID(media_id)


# TODO: replace models with more specific variants from .schema package when defined
class Media(BaseIDNameModel):
    """Model representing media information"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "b0feaabc-da2b-442e-9b3e-55c11c2c2ff2",
                "name": "cat.jpg",
            },
        }
    }


class PaginationMetadata(BaseModel):
    """Pagination metadata for list responses"""

    total_count: int
    page: int
    page_size: int
    total_pages: int


class MediaListResponse(BaseModel):
    """Response model for paginated media list"""

    data: list[Media]
    pagination: PaginationMetadata


@router.get(
    "",
    response_model=MediaListResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "List of available media",
            "content": {
                "application/json": {
                    "examples": MEDIA_RESPONSE_LIST_EXAMPLES,
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid query parameters"},
    },
)
async def list_media(
    page: int = Query(default=1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Number of items per page (max 100)"),
    start_date: date | None = Query(
        default=None, description="Filter media created on or after this date (YYYY-MM-DD)"
    ),
    end_date: date | None = Query(default=None, description="Filter media created on or before this date (YYYY-MM-DD)"),
) -> list[Media]:
    """List available media items with pagination and optional date filtering"""
    _ = page, page_size
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start date cannot be after end date",
        )
    raise NotImplementedError


@router.get(
    "/{media_id}",
    response_model=Media,
    responses={
        status.HTTP_200_OK: {"description": "Media found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_media(
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> Media:
    """Get detailed information about a specific media item"""
    _ = media_id
    raise NotImplementedError


@router.get(
    "/{media_id}/full",
    responses={
        status.HTTP_200_OK: {
            "description": "Media found",
            "content": {  # TODO: Indicate all supported binary formats
                "image/jpeg": {"example": "...binary data..."},
                "image/png": {"example": "...binary data..."},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_full(
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:
    # In Geti Classic media microservice, image was streamed from object storage, so `StreamingResponse` is appropriate.
    # Now, since binaries will be stored on the local file system, `FileResponse` is more suitable.
    """Get the media file at full resolution"""
    _ = media_id
    raise NotImplementedError


@router.get(
    "/{media_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Media found", "content": {"image/jpeg": {}}},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_thumbnail(
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:  # If thumbnails are stored in SQLite as BLOB, `StreamingResponse` would be appropriate.
    """Get a thumbnail preview of the media item"""
    _ = media_id
    raise NotImplementedError


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=Media,
    responses={
        status.HTTP_201_CREATED: {"description": "Media successfully uploaded"},
    },
)
async def upload_media(file: Annotated[UploadFile, File(description="Media file to upload")]) -> Media:
    """Upload a new media file to the system"""
    _ = file
    raise NotImplementedError


@router.delete(
    "/{media_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media successfully deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def delete_media(
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> None:
    """
    Permanently delete a media item from the system.

    Removes both the media file and its associated metadata, including the thumbnail.
    This operation cannot be undone.
    """
    _ = media_id
    raise NotImplementedError
