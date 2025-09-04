# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.responses import StreamingResponse

from app.api.dependencies import get_dataset_item_id, get_dataset_item_service
from app.schemas import DatasetItem, DatasetItemsWithPagination
from app.schemas.base import Pagination
from app.services import DatasetItemService, ResourceNotFoundError

router = APIRouter(prefix="/api/dataset/items", tags=["Dataset items"])

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_201_CREATED: {"description": "Dataset item created", "model": DatasetItem}},
)
async def create_dataset_item(
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
    file: Annotated[UploadFile, File()],
) -> DatasetItem:
    """Create a new dataset item"""
    return dataset_item_service.create_dataset_item(file.file)


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset items", "model": DatasetItemsWithPagination},
    },
)
async def list_dataset_items(
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_ITEMS_NUMBER_RETURNED)] = DEFAULT_DATASET_ITEMS_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0)] = 0,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
) -> DatasetItemsWithPagination:
    """Get information about available dataset items"""
    dataset_items = dataset_item_service.list_dataset_items(
        limit=limit, offset=offset, start_date=start_date, end_date=end_date
    )
    return DatasetItemsWithPagination(  # TODO: implement
        items=dataset_items,
        pagination=Pagination(
            limit=limit,
            offset=offset,
            total=0,
            count=len(dataset_items),
        ),
    )


@router.get(
    "/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItem},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item not found"},
    },
)
async def get_dataset_item(
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
) -> DatasetItem:
    """Get information about a specific dataset item"""
    try:
        return dataset_item_service.get_dataset_item_by_id(dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{dataset_item_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item binary not found"},
    },
)
async def get_dataset_item_binary(
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
) -> StreamingResponse:
    """Get dataset item binary content"""
    try:
        dataset_item_binary = dataset_item_service.get_dataset_item_binary_by_id(dataset_item_id)
        return StreamingResponse(content=dataset_item_binary)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{dataset_item_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item thumbnail not found"},
    },
)
async def get_dataset_item_thumbnail(
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
) -> StreamingResponse:
    """Get dataset item thumbnail binary content"""
    try:
        dataset_item_thumbnail = dataset_item_service.get_dataset_item_thumbnail_by_id(dataset_item_id)
        return StreamingResponse(content=dataset_item_thumbnail)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item not found"},
    },
)
async def delete_dataset_item(
    dataset_item_id: Annotated[UUID, Depends(get_dataset_item_id)],
    dataset_item_service: Annotated[DatasetItemService, Depends(get_dataset_item_service)],
) -> None:
    """Delete dataset item content"""
    try:
        dataset_item_service.delete_dataset_item(dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
