from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse

from app.api.dependencies import is_valid_uuid
from app.schemas.base import BaseIDNameModel

router = APIRouter(prefix="/api/media", tags=["Media"])


def get_media_id(media_id: str) -> str:
    """Initializes and validates a media ID"""
    if not is_valid_uuid(media_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid media ID")
    return media_id


# TODO: replace by a more specific model from .schema package
class Media(BaseIDNameModel):
    """Model representing media information"""


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available media", "model": list[Media]},
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
    """List the available media"""
    _ = page, page_size, start_date, end_date
    raise NotImplementedError


@router.get(
    "/{media_id}",
    responses={
        status.HTTP_200_OK: {"description": "Media found", "model": Media},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_media(
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> Media:
    """Get info about a media item"""
    _ = media_id
    raise NotImplementedError


@router.get(
    "/{media_id}/full",
    responses={
        status.HTTP_200_OK: {
            "description": "Media found",
            "content": {  # TODO: Indicate all supported binary formats
                "image/jpeg": {},
                "image/png": {},
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
    """Get the media binary data with full resolution"""
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
    """Get a thumbnail of the media item"""
    _ = media_id
    raise NotImplementedError


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Media successfully uploaded", "model": Media},
    },
)
async def upload_media(file: Annotated[UploadFile, File(description="Media file to upload")]) -> Media:
    """Upload a new media item"""
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
    """Delete a media item"""
    _ = media_id
    raise NotImplementedError
