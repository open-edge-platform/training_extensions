# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from starlette.responses import FileResponse

from app.api.dependencies import get_dataset_revision, get_dataset_revision_service, get_project
from app.api.schemas.dataset_item import (
    DatasetItemRevisionView,
    DatasetItemsRevisionWithPagination,
    DatasetItemsWithPagination,
)
from app.api.validators import DatasetItemID, DatasetRevisionID
from app.core.models import Pagination
from app.models import DatasetItemFormat, DatasetItemSubset, Project
from app.models.dataset_revision import DatasetRevision
from app.services import DatasetRevisionService

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
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> DatasetItemsRevisionWithPagination:
    """List the items in a dataset revision. This endpoint supports pagination."""
    items_data, total_count = dataset_revision_service.list_dataset_revision_items(
        project_id=project.id,
        dataset_revision=dataset_revision,
        limit=limit,
        offset=offset,
        subset=subset,
    )

    items = []
    for item_data in items_data:
        item_view = DatasetItemRevisionView(
            id=UUID(item_data["id"]),
            name=Path(item_data["image"]).stem,
            format=DatasetItemFormat(Path(item_data["image"]).suffix.lstrip(".")),
            width=item_data["image_info"]["width"],
            height=item_data["image_info"]["height"],
            subset=DatasetItemSubset(item_data["subset"]),
        )
        items.append(item_view)

    return DatasetItemsRevisionWithPagination(
        items=items,
        pagination=Pagination(
            total=total_count,
            limit=limit,
            offset=offset,
            count=len(items),
        ),
    )


@router.get(
    "/items/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemRevisionView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, revision, or project not found"},
    },
)
def get_dataset_revision_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> DatasetItemRevisionView:
    """Get information about a specific item in the dataset revision"""
    item_data = dataset_revision_service.get_dataset_revision_item(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    )

    return DatasetItemRevisionView(
        id=UUID(item_data["id"]),
        name=Path(item_data["image"]).stem,
        format=DatasetItemFormat(Path(item_data["image"]).suffix.lstrip(".")),
        width=item_data["image_info"]["width"],
        height=item_data["image_info"]["height"],
        subset=DatasetItemSubset(item_data["subset"]),
    )


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
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> FileResponse:
    """Get the image data of an item in the dataset revision"""
    binary_path = dataset_revision_service.get_dataset_revision_item_binary_path(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    )
    return FileResponse(binary_path, media_type="application/octet-stream")


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
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> FileResponse:
    """Get the thumbnail of an item in the dataset revision"""
    # TODO: correctly compute thumbnail image and return it.
    binary_path = dataset_revision_service.get_dataset_revision_item_binary_path(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    )
    return FileResponse(binary_path, media_type="image/jpeg")


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
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> None:
    """Delete the files associated with a dataset revision"""
    dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=dataset_revision_id)
