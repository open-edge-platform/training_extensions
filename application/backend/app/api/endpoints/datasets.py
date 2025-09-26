# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.openapi.models import Example
from starlette.responses import FileResponse

from app.api.dependencies import (
    get_dataset_item_id,
    get_dataset_service,
    get_file_name_and_extension,
    get_file_size,
    get_project_id,
)
from app.schemas import DatasetItem, DatasetItemsWithPagination
from app.schemas.base import Pagination
from app.schemas.dataset_item import DatasetItemAnnotation, DatasetItemAnnotations, DatasetItemAnnotationsWithSource
from app.services import DatasetService, ResourceNotFoundError
from app.services.dataset_service import AnnotationValidationError, InvalidImageError

router = APIRouter(prefix="/api/projects/{project_id}/dataset/items", tags=["Datasets"])

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100

SET_DATASET_ITEM_ANNOTATIONS_BODY_EXAMPLES = {
    "full_image": Example(
        summary="Full image annotation",
        value={
            "annotations": [
                {"labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}], "shape": {"type": "full_image"}}
            ],
        },
    ),
    "rectangle": Example(
        summary="Rectangle annotation",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                }
            ],
        },
    ),
    "polygon": Example(
        summary="Polygon annotation",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "polygon", "points": [[10, 20], [20, 60], [30, 40]]},
                }
            ],
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_201_CREATED: {"description": "Dataset item created", "model": DatasetItem}},
)
def add_dataset_item(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    file_name_and_extension: Annotated[tuple[str, str], Depends(get_file_name_and_extension)],
    size: Annotated[int, Depends(get_file_size)],
    file: Annotated[UploadFile, File()],
) -> DatasetItem:
    """Add a new item to the dataset by uploading an image"""
    name, format = file_name_and_extension
    try:
        return dataset_service.create_dataset_item(
            project_id=project_id,
            file=file.file,
            name=name,
            format=format,
            size=size,
            user_reviewed=True,
        )
    except InvalidImageError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid image has been uploaded.")


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset items", "model": DatasetItemsWithPagination},
    },
)
def list_dataset_items(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """List the available dataset items and their metadata. This endpoint supports pagination."""
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Start date must be before end date."
        )
    total = dataset_service.count_dataset_items(project_id=project_id, start_date=start_date, end_date=end_date)
    dataset_items = dataset_service.list_dataset_items(
        project_id=project_id, limit=limit, offset=offset, start_date=start_date, end_date=end_date
    )
    return DatasetItemsWithPagination(
        items=dataset_items,
        pagination=Pagination(
            limit=limit,
            offset=offset,
            total=total,
            count=len(dataset_items),
        ),
    )


@router.get(
    "/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItem},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def get_dataset_item(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItem:
    """Get information about a specific dataset item"""
    try:
        return dataset_service.get_dataset_item_by_id(project_id=project_id, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{dataset_item_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, dataset item binary or project not found"},
    },
)
def get_dataset_item_binary(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get dataset item binary content"""
    try:
        binary_path = dataset_service.get_dataset_item_binary_path_by_id(
            project_id=project_id, dataset_item_id=dataset_item_id
        )
        return FileResponse(path=binary_path)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{dataset_item_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, dataset item thumbnail or project not found"},
    },
)
def get_dataset_item_thumbnail(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get dataset item thumbnail binary content"""
    try:
        thumbnail_path = dataset_service.get_dataset_item_thumbnail_path_by_id(
            project_id=project_id, dataset_item_id=dataset_item_id
        )
        return FileResponse(path=thumbnail_path)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{dataset_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Dataset item deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def delete_dataset_item(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete an item from the dataset"""
    try:
        dataset_service.delete_dataset_item(project_id=project_id, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{dataset_item_id}/annotations",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Dataset item annotation created", "model": DatasetItemAnnotation},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or invalid annotation content"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def set_dataset_item_annotations(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_item_annotations: Annotated[
        DatasetItemAnnotations, Body(openapi_examples=SET_DATASET_ITEM_ANNOTATIONS_BODY_EXAMPLES)
    ],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemAnnotationsWithSource:
    """Set dataset item annotations"""
    try:
        return dataset_service.set_dataset_item_annotations(
            project_id=project_id, dataset_item_id=dataset_item_id, annotations=dataset_item_annotations.annotations
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AnnotationValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{dataset_item_id}/annotations",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemAnnotationsWithSource},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def get_dataset_item_annotations(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemAnnotationsWithSource:
    """Get the dataset item annotations"""
    try:
        return dataset_service.get_dataset_item_annotations(project_id=project_id, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{dataset_item_id}/annotations",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Dataset item annotations deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def delete_dataset_item_annotation(
    project_id: Annotated[UUID, Depends(get_project_id)],
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete dataset item annotations"""
    try:
        dataset_service.delete_dataset_item_annotations(project_id=project_id, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
