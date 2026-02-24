# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from io import BytesIO
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.openapi.models import Example
from PIL.Image import Image
from starlette.responses import FileResponse, StreamingResponse

from app.api.dependencies import get_dataset_service, get_file_name_and_extension, get_media_service, get_project
from app.api.schemas.media import (
    AnnotatedVideoFrame,
    MediaAnnotations,
    MediaView,
    MediaViewAdapter,
    MediaWithPagination,
    SetMediaAnnotations,
)
from app.api.validators import MediaID
from app.core.models import Pagination
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Project
from app.models.media import ImageFormat, MediaType, VideoFormat
from app.services import DatasetService, MediaService
from app.services.dataset_service import AnnotationValidationError
from app.services.media_service import InvalidImageError, MediaFilters

router = APIRouter(prefix="/api/projects/{project_id}/dataset/media", tags=["Media"])

DEFAULT_MEDIA_NUMBER_RETURNED = 10
MAX_MEDIA_NUMBER_RETURNED = 100

DEFAULT_FRAME_INDEX_FROM = 0
DEFAULT_FRAME_INDEX_TO = 50

SET_MEDIA_ANNOTATIONS_BODY_EXAMPLES = {
    "single_label": Example(
        summary="Single label (multiclass classification)",
        value={
            "annotations": [
                {"labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}], "shape": {"type": "full_image"}}
            ],
        },
    ),
    "multi_label": Example(
        summary="Multiple labels (multilabel classification)",
        value={
            "annotations": [
                {
                    "labels": [
                        {"id": "d476573e-d43c-42a6-9327-199a9aa75c33"},
                        {"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"},
                    ],
                    "shape": {"type": "full_image"},
                },
            ],
        },
    ),
    "bounding_boxes": Example(
        summary="Bounding boxes (detection)",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                },
                {
                    "labels": [{"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"}],
                    "shape": {"type": "rectangle", "x": 150, "y": 250, "width": 80, "height": 120},
                },
            ],
        },
    ),
    "polygons": Example(
        summary="Polygons (segmentation)",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "polygon", "points": [[10, 20], [20, 60], [30, 40]]},
                },
                {
                    "labels": [{"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"}],
                    "shape": {"type": "polygon", "points": [[150, 250], [180, 300], [200, 280]]},
                },
            ],
        },
    ),
    "empty": Example(
        summary="No objects (detection / segmentation)",
        value={
            "annotations": [],
        },
    ),
}


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


def _write_image_to_response(image: Image, filename: str, cache_control: str | None = None) -> StreamingResponse:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    headers = {"Content-Disposition": f"inline; filename={filename}"}
    if cache_control:
        headers["Cache-Control"] = cache_control
    return StreamingResponse(buffer, media_type="image/jpeg", headers=headers)


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
        return MediaViewAdapter.validate_python(media, from_attributes=True)
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
        exclude_types=[MediaType.VIDEO_FRAME],
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
        exclude_types=[MediaType.VIDEO_FRAME],
    )
    return MediaWithPagination(
        items=[MediaViewAdapter.validate_python(media, from_attributes=True) for media in media_list],
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
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    return MediaViewAdapter.validate_python(media, from_attributes=True)


@router.get(
    "/{media_id}/frames",
    responses={
        status.HTTP_200_OK: {"description": "List of annotated video frames", "model": list[AnnotatedVideoFrame]},
        status.HTTP_404_NOT_FOUND: {"description": "Project or video not found"},
    },
)
def list_video_frames(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    frame_index_from: Annotated[int, Query(ge=0)] = DEFAULT_FRAME_INDEX_FROM,
    frame_index_to: Annotated[int, Query(ge=0)] = DEFAULT_FRAME_INDEX_TO,
) -> list[AnnotatedVideoFrame]:
    """Lists annotated video frames with frame index range"""
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type != MediaType.VIDEO:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Requested media is not video.")
    annotated_video_frames = media_service.list_annotated_video_frames_by_video_id(
        project=project,
        video_id=media_id,
        frame_index_from=frame_index_from,
        frame_index_to=frame_index_to,
    )
    return [
        AnnotatedVideoFrame(
            media_id=video_frame.id,
            frame_index=video_frame.frame_index,
            dataset=MediaAnnotations(
                annotations=dataset_item.annotation_data,  # type: ignore[arg-type]
                prediction_model_id=dataset_item.prediction_model_id,
                user_reviewed=dataset_item.user_reviewed,
            ),
        )
        for (dataset_item, video_frame) in annotated_video_frames
    ]


@router.get(
    "/{media_id}/binary",
    response_model=None,
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
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> FileResponse | StreamingResponse:
    """Get media binary content"""
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    print(media)
    if media.type == MediaType.VIDEO and frame_index is not None:
        # Video frames can be identified by video ID and frame index
        if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
            )
        video_frame = media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=media_id, frame_index=frame_index
        )
        if video_frame is None:
            frame_binary = media_service.get_frame_binary(project=project, video=media, frame_index=frame_index)
            return _write_image_to_response(image=frame_binary, filename=f"{media.name}_frame_{frame_index}.jpeg")
        media = video_frame

    binary_path = media_service.get_media_binary_path_by_id(project_id=project.id, media_id=media.id)
    filename = f"{media.name}.{media.format.value.lower()}"
    return FileResponse(path=binary_path, filename=filename)


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
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> StreamingResponse:
    """Get media thumbnail binary content"""
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type == MediaType.VIDEO and frame_index is not None:
        # Video frames can be identified by video ID and frame index
        if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
            )
        video_frame = media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=media_id, frame_index=frame_index
        )
        if video_frame is None:
            frame_binary = media_service.get_frame_thumbnail(project=project, video=media, frame_index=frame_index)
            return _write_image_to_response(image=frame_binary, filename=f"{media.name}_frame_{frame_index}_thumb.jpeg")
        media = video_frame

    thumbnail = media_service.generate_media_thumbnail(project=project, media=media)
    return _write_image_to_response(
        image=thumbnail, filename=f"{media_id}_thumb.jpeg", cache_control="public, max-age=31536000"
    )


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
    media_service.delete_media(project=project, media_id=media_id)


@router.post(
    "/{media_id}/annotations",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Annotation created or updated", "model": MediaAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or invalid annotation content"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
    },
)
def set_media_annotations(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_annotations: Annotated[SetMediaAnnotations, Body(openapi_examples=SET_MEDIA_ANNOTATIONS_BODY_EXAMPLES)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> MediaAnnotations:
    """Set media annotations"""
    # Dataset item has the same ID as media
    dataset_item_id = media_id
    try:
        media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
        if media.type == MediaType.VIDEO:
            # Video frames can be identified by video ID and frame index
            if frame_index is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided."
                )
            if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
                )
            video_frame = media_service.get_video_frame_by_video_id_and_index(
                project=project, video_id=media_id, frame_index=frame_index
            )
            if video_frame is None:
                video_frame = media_service.extract_video_frame(project=project, video=media, frame_index=frame_index)
                dataset_service.create_dataset_item(
                    project=project,
                    media=video_frame,
                    user_reviewed=True,
                )
            dataset_item_id = video_frame.id

        dataset_item = dataset_service.set_dataset_item_annotations(
            project=project,
            dataset_item_id=dataset_item_id,
            annotations=media_annotations.annotations,
            # Annotations submitted via API are considered user-reviewed, unlike auto-generated predictions
            user_reviewed=True,
        )
        return MediaAnnotations(
            annotations=dataset_item.annotation_data,  # type: ignore[arg-type]
            prediction_model_id=dataset_item.prediction_model_id,
            user_reviewed=dataset_item.user_reviewed,
        )
    except AnnotationValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{media_id}/annotations",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Annotation found", "model": MediaAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {
            "description": "Media, dataset item or project not found or media is not annotated"
        },
    },
)
def get_media_annotations(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> MediaAnnotations:
    """Get the media annotations"""
    # Dataset item has the same ID as media
    dataset_item_id = media_id
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type == MediaType.VIDEO:
        # Video frames can be identified by video ID and frame index
        if frame_index is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
        if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
            )
        video_frame = media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=media_id, frame_index=frame_index
        )
        if video_frame is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Video frame not found for the given index."
            )
        dataset_item_id = video_frame.id

    dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
    if dataset_item.annotation_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media has not been annotated yet.")
    return MediaAnnotations(
        annotations=dataset_item.annotation_data,
        prediction_model_id=dataset_item.prediction_model_id,
        user_reviewed=dataset_item.user_reviewed,
    )


@router.delete(
    "/{media_id}/annotations",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media annotations deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
    },
)
def delete_media_annotation(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> None:
    """Delete media annotations"""
    # Dataset item has the same ID as media
    dataset_item_id = media_id
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type == MediaType.VIDEO:
        # Video frames can be identified by video ID and frame index
        if frame_index is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
        if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
            )
        video_frame = media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=media_id, frame_index=frame_index
        )
        if video_frame is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Video frame not found for the given index."
            )
        dataset_item_id = video_frame.id

    dataset_service.delete_dataset_item_annotations(project=project, dataset_item_id=dataset_item_id)
