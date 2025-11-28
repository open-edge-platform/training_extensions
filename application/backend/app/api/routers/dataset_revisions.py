# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.responses import FileResponse

from app.api.dependencies import get_dataset_service, get_project
from app.api.schemas.dataset_item import DatasetItemsWithPagination, DatasetItemView
from app.api.validators import DatasetItemID, DatasetRevisionID
from app.core.models import Pagination
from app.models import DatasetItemSubset, Project
from app.services import DatasetService, ResourceNotFoundError
from app.services.dataset_service import DatasetItemFilters

router = APIRouter(
    prefix="/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}",
    tags=["Dataset Revisions"],
)

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100


@router.get(
    "/items",
    responses={
        status.HTTP_200_OK: {
            "description": "List of dataset items in the revision",
            "model": DatasetItemsWithPagination,
        },
        status.HTTP_404_NOT_FOUND: {"description": "Dataset revision or project not found"},
    },
)
def list_dataset_revision_items(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_id: DatasetRevisionID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """List the items in a dataset revision. This endpoint supports pagination."""
    try:
        # Verify the revision exists
        dataset_service.get_dataset_revision(project_id=project.id, revision_id=dataset_revision_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    total = dataset_service.count_dataset_items(
        project=project,
        subset=subset,
    )
    dataset_items = dataset_service.list_dataset_items(
        project_id=project.id,
        filters=DatasetItemFilters(
            limit=limit,
            offset=offset,
            subset=subset,
        ),
    )
    return DatasetItemsWithPagination(
        items=[DatasetItemView.model_validate(dataset_item, from_attributes=True) for dataset_item in dataset_items],
        pagination=Pagination(
            limit=limit,
            offset=offset,
            total=total,
            count=len(dataset_items),
        ),
    )


@router.get(
    "/items/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, revision, or project not found"},
    },
)
def get_dataset_revision_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_id: DatasetRevisionID,
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemView:
    """Get information about a specific item in the dataset revision"""
    try:
        # Verify the revision exists
        dataset_service.get_dataset_revision(project_id=project.id, revision_id=dataset_revision_id)
        # Get the dataset item
        dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
        return DatasetItemView.model_validate(dataset_item, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/items/{dataset_item_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, binary, revision, or project not found"},
    },
)
def get_dataset_revision_item_binary(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_id: DatasetRevisionID,
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the image data of an item in the dataset revision"""
    try:
        # Verify the revision exists
        dataset_service.get_dataset_revision(project_id=project.id, revision_id=dataset_revision_id)
        # Get the binary path
        binary_path = dataset_service.get_dataset_item_binary_path_by_id(
            project_id=project.id, dataset_item_id=dataset_item_id
        )
        return FileResponse(path=binary_path)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/items/{dataset_item_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, thumbnail, revision, or project not found"},
    },
)
def get_dataset_revision_item_thumbnail(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_id: DatasetRevisionID,
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the thumbnail of an item in the dataset revision"""
    try:
        # Verify the revision exists
        dataset_service.get_dataset_revision(project_id=project.id, revision_id=dataset_revision_id)
        # Get the thumbnail path
        thumbnail_path = dataset_service.get_dataset_item_thumbnail_path_by_id(
            project=project, dataset_item_id=dataset_item_id
        )
        return FileResponse(path=thumbnail_path)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Dataset revision files deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset revision or project not found"},
    },
)
def delete_dataset_revision_files(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_id: DatasetRevisionID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete the files associated with a dataset revision"""
    try:
        dataset_service.delete_dataset_revision_files(project_id=project.id, revision_id=dataset_revision_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
