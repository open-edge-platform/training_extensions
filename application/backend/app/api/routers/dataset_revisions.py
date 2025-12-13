# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from starlette.responses import FileResponse

from app.api.dependencies import get_dataset_revision, get_dataset_service, get_project
from app.api.schemas.dataset_item import DatasetItemsWithPagination, DatasetItemView
from app.api.validators import DatasetItemID, DatasetRevisionID
from app.models import DatasetItemSubset, Project
from app.models.dataset_revision import DatasetRevision
from app.services import DatasetService

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
    _project: Annotated[Project, Depends(get_project)],
    _dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    _limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    _offset: Annotated[int, Query(ge=0)] = 0,
    _subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """List the items in a dataset revision. This endpoint supports pagination."""
    raise NotImplementedError


@router.get(
    "/items/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, revision, or project not found"},
    },
)
def get_dataset_revision_item(
    _project: Annotated[Project, Depends(get_project)],
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    _dataset_item_id: DatasetItemID,
    _dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemView:
    """Get information about a specific item in the dataset revision"""
    raise NotImplementedError


@router.get(
    "/items/{dataset_item_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, binary, revision, or project not found"},
    },
)
def get_dataset_revision_item_binary(
    _project: Annotated[Project, Depends(get_project)],
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    _dataset_item_id: DatasetItemID,
    _dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the image data of an item in the dataset revision"""
    raise NotImplementedError


@router.get(
    "/items/{dataset_item_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, thumbnail, revision, or project not found"},
    },
)
def get_dataset_revision_item_thumbnail(
    _project: Annotated[Project, Depends(get_project)],
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    _dataset_item_id: DatasetItemID,
    _dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the thumbnail of an item in the dataset revision"""
    raise NotImplementedError


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
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete the files associated with a dataset revision"""
    dataset_service.delete_dataset_revision_files(project_id=project.id, revision_id=dataset_revision_id)
