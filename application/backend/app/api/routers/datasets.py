# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_dataset_service, get_project
from app.api.schemas.dataset import DatasetStatisticsView
from app.api.schemas.dataset_item import DatasetItemsWithPagination, DatasetItemView
from app.api.validators import DatasetItemID, ProjectID, normalize_datetime_to_utc
from app.core.models import Pagination
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Project
from app.services import DatasetService
from app.services.dataset_service import DatasetItemFilters

router = APIRouter(prefix="/api/projects/{project_id}/dataset", tags=["Datasets"])

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100


@router.get(
    "/items",
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset items", "model": DatasetItemsWithPagination},
    },
)
def list_dataset_items(  # noqa: PLR0913
    project: Annotated[Project, Depends(get_project)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
    annotation_status: Annotated[DatasetItemAnnotationStatus | None, Query()] = None,
    labels: Annotated[list[UUID] | None, Query()] = None,
    subset: Annotated[DatasetItemSubset | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """List the available dataset items and their metadata. This endpoint supports pagination."""
    start_date = normalize_datetime_to_utc(start_date)
    end_date = normalize_datetime_to_utc(end_date)

    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Start date must be before end date."
        )
    total = dataset_service.count_dataset_items(
        project=project,
        start_date=start_date,
        end_date=end_date,
        annotation_status=annotation_status,
        label_ids=labels,
        subset=subset,
    )
    dataset_items = dataset_service.list_dataset_items(
        project_id=project.id,
        filters=DatasetItemFilters(
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=labels,
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
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def get_dataset_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemView:
    """Get information about a specific dataset item"""
    dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
    return DatasetItemView.model_validate(dataset_item, from_attributes=True)


@router.get(
    "/statistics",
    responses={
        status.HTTP_200_OK: {"description": "Dataset statistics retrieved", "model": DatasetStatisticsView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_dataset_statistics(
    project_id: ProjectID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetStatisticsView:
    """Get statistics about the number of media and annotations in a dataset"""
    dataset_statistics = dataset_service.get_dataset_statistics(project_id=project_id)
    return DatasetStatisticsView.model_validate(dataset_statistics, from_attributes=True)
