# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from starlette.responses import FileResponse

from app.api.dependencies import get_dataset_revision, get_dataset_service, get_project
from app.api.schemas.dataset_item import DatasetItemsWithPagination, DatasetItemView
from app.api.validators import DatasetItemID, DatasetRevisionID
from app.core.models import Pagination
from app.models import DatasetItemFormat, DatasetItemSubset, Project
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
    project: Annotated[Project, Depends(get_project)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """List the items in a dataset revision. This endpoint supports pagination."""
    items_data, total_count = dataset_service.list_dataset_revision_items(
        project_id=project.id,
        revision_id=dataset_revision.id,
        limit=limit,
        offset=offset,
        subset=subset,
    )

    # Convert items to DatasetItemView format
    items = []
    for item_data in items_data:
        # Extract relevant fields from the parquet data
        # Note: The exact field mapping depends on the datumaro export schema
        item_view = DatasetItemView(
            id=item_data.get("id", item_data.get("image", "unknown")),
            name=Path(item_data.get("image", "")).stem if isinstance(item_data.get("image"), str) else "unknown",
            format=DatasetItemFormat.JPG,  # Default, could be extracted from image path
            width=item_data.get("image_info", {}).get("width", 0)
            if isinstance(item_data.get("image_info"), dict)
            else 0,
            height=item_data.get("image_info", {}).get("height", 0)
            if isinstance(item_data.get("image_info"), dict)
            else 0,
            size=0,  # Not available in parquet
            source_id=None,
            subset=DatasetItemSubset(item_data["subset"]) if "subset" in item_data else DatasetItemSubset.UNASSIGNED,
        )
        items.append(item_view)

    return DatasetItemsWithPagination(
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
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, revision, or project not found"},
    },
)
def get_dataset_revision_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemView:
    """Get information about a specific item in the dataset revision"""
    item_data = dataset_service.get_dataset_revision_item(
        project_id=project.id,
        revision_id=dataset_revision.id,
        item_id=str(dataset_item_id),
    )

    # Convert to DatasetItemView format
    return DatasetItemView(
        id=item_data.get("id", item_data.get("image", "unknown")),
        name=Path(item_data.get("image", "")).stem if isinstance(item_data.get("image"), str) else "unknown",
        format=DatasetItemFormat.JPG,  # Default, could be extracted from image path
        width=item_data.get("image_info", {}).get("width", 0) if isinstance(item_data.get("image_info"), dict) else 0,
        height=item_data.get("image_info", {}).get("height", 0) if isinstance(item_data.get("image_info"), dict) else 0,
        size=0,  # Not available in parquet
        source_id=None,
        subset=DatasetItemSubset(item_data["subset"]) if "subset" in item_data else DatasetItemSubset.UNASSIGNED,
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
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the image data of an item in the dataset revision"""
    binary_path = dataset_service.get_dataset_revision_item_binary_path(
        project_id=project.id,
        revision_id=dataset_revision.id,
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
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get the thumbnail of an item in the dataset revision"""
    binary_path = dataset_service.get_dataset_revision_item_binary_path(
        project_id=project.id,
        revision_id=dataset_revision.id,
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
    _dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete the files associated with a dataset revision"""
    dataset_service.delete_dataset_revision_files(project_id=project.id, revision_id=dataset_revision_id)
