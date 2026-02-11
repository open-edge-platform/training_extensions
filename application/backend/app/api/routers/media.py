# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from io import BytesIO
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.responses import FileResponse, StreamingResponse

from app.api.dependencies import get_dataset_service, get_file_name_and_extension, get_media_service, get_project
from app.api.schemas.media import MediaView, MediaWithPagination
from app.api.validators import MediaID
from app.core.models import Pagination
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Project
from app.models.media import ImageFormat, VideoFormat
from app.services import DatasetService, MediaService, ResourceNotFoundError
from app.services.media_service import InvalidImageError, MediaFilters

router = APIRouter(prefix="/api/projects/{project_id}/dataset/media", tags=["Media"])

DEFAULT_MEDIA_NUMBER_RETURNED = 10
MAX_MEDIA_NUMBER_RETURNED = 100


def _parse_media_format(extension: str) -> ImageFormat | VideoFormat:
    try:
        return ImageFormat[extension.upper()]
    except KeyError:
        try:
            return VideoFormat[extension.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unsupported media extension: {extension}",
            )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=MediaView,
    responses={
        status.HTTP_201_CREATED: {"description": "Media created"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid media has been uploaded"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def add_media(
    project: Annotated[Project, Depends(get_project)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    file_name_and_extension: Annotated[tuple[str, str], Depends(get_file_name_and_extension)],
    file: Annotated[UploadFile, File()],
) -> MediaView:
    """Add a new media to the dataset by uploading an image or a video"""
    name, extension = file_name_and_extension
    format = _parse_media_format(extension)
    try:
        if isinstance(format, ImageFormat):
            media = media_service.create_image(
                project=project,
                data=file.file,
                name=name,
                format=format,
            )
            dataset_service.create_dataset_item(
                project=project,
                media=media,
                user_reviewed=False,
            )
        else:
            # Dataset items for videos are created separately after video upload for each frame being annotated
            media = media_service.create_video(
                project=project,
                data=file.file,
                name=name,
                format=format,
            )
        return MediaView.model_validate(media, from_attributes=True)
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid image has been uploaded."
        )


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available media", "model": MediaWithPagination},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_media(  # noqa: PLR0913
    project: Annotated[Project, Depends(get_project)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_MEDIA_NUMBER_RETURNED)] = DEFAULT_MEDIA_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
    annotation_status: Annotated[DatasetItemAnnotationStatus | None, Query()] = None,
    labels: Annotated[list[UUID] | None, Query()] = None,
    subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> MediaWithPagination:
    """List the available media and their metadata. This endpoint supports pagination."""
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Start date must be before end date."
        )
    total = media_service.count_media(
        project=project,
        start_date=start_date,
        end_date=end_date,
        annotation_status=annotation_status,
        label_ids=labels,
        subset=subset,
    )
    media_list = media_service.list_media(
        project_id=project.id,
        filters=MediaFilters(
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=labels,
            subset=subset,
        ),
    )
    return MediaWithPagination(
        items=[MediaView.model_validate(media, from_attributes=True) for media in media_list],
        pagination=Pagination(
            limit=limit,
            offset=offset,
            total=total,
            count=len(media_list),
        ),
    )


@router.get(
    "/{media_id}",
    responses={
        status.HTTP_200_OK: {"description": "Media item found", "model": MediaView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media or project not found"},
    },
)
def get_media(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> MediaView:
    """Get information about a specific media"""
    try:
        media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
        return MediaView.model_validate(media, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{media_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Media binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, media binary or project not found"},
    },
)
def get_media_binary(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> FileResponse:
    """Get media binary content"""
    try:
        binary_path = media_service.get_media_binary_path_by_id(project_id=project.id, media_id=media_id)
        media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
        filename = f"{media.name}.{media.format.value.lower()}"
        return FileResponse(path=binary_path, filename=filename)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{media_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Media thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, media thumbnail or project not found"},
    },
)
def get_media_thumbnail(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> StreamingResponse:
    """Get media thumbnail binary content"""
    try:
        thumbnail = media_service.generate_media_thumbnail(project=project, media_id=media_id)
        buffer = BytesIO()
        thumbnail.save(buffer, format="JPEG")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename={media_id}.jpeg",
                "Cache-Control": "public, max-age=31536000",
            },
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{media_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media or project not found"},
    },
)
def delete_media(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> None:
    """Delete media from the dataset"""
    try:
        media_service.delete_media(project=project, media_id=media_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
