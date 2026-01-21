# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from io import BytesIO
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.openapi.models import Example
from starlette.responses import FileResponse, StreamingResponse

from app.api.dependencies import get_dataset_service, get_file_name_and_extension, get_project
from app.api.schemas.dataset_item import (
    DatasetItemAnnotations,
    DatasetItemAssignSubset,
    DatasetItemsWithPagination,
    DatasetItemView,
    SetDatasetItemAnnotations,
)
from app.api.validators import DatasetItemID
from app.core.models import Pagination
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Project
from app.services import DatasetService, ResourceNotFoundError
from app.services.dataset_service import (
    AnnotationValidationError,
    DatasetItemFilters,
    InvalidImageError,
    SubsetAlreadyAssignedError,
)

router = APIRouter(prefix="/api/projects/{project_id}/dataset/items", tags=["Datasets"])

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100

SET_DATASET_ITEM_ANNOTATIONS_BODY_EXAMPLES = {
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


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DatasetItemView,
    responses={
        status.HTTP_201_CREATED: {"description": "Dataset item created"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid image has been uploaded"},
    },
)
def add_dataset_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    file_name_and_extension: Annotated[tuple[str, str], Depends(get_file_name_and_extension)],
    file: Annotated[UploadFile, File()],
) -> DatasetItemView:
    """Add a new item to the dataset by uploading an image"""
    name, format = file_name_and_extension
    try:
        dataset_item = dataset_service.create_dataset_item(
            project=project,
            data=file.file,
            name=name,
            format=format,
            user_reviewed=True,
        )
        return DatasetItemView.model_validate(dataset_item, from_attributes=True)
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid image has been uploaded."
        )


@router.get(
    "",
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
    "/{dataset_item_id}",
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
    try:
        dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
        return DatasetItemView.model_validate(dataset_item, from_attributes=True)
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
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get dataset item binary content"""
    try:
        binary_path = dataset_service.get_dataset_item_binary_path_by_id(
            project_id=project.id, dataset_item_id=dataset_item_id
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
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> StreamingResponse:
    """Get dataset item thumbnail binary content"""
    try:
        thumbnail = dataset_service.generate_dataset_item_thumbnail(project=project, dataset_item_id=dataset_item_id)
        buffer = BytesIO()
        thumbnail.save(buffer, format="JPEG")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename={dataset_item_id}.jpeg",
                "Cache-Control": "public, max-age=31536000",
            },
        )
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
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete an item from the dataset"""
    try:
        dataset_service.delete_dataset_item(project=project, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{dataset_item_id}/annotations",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Dataset item annotation created", "model": DatasetItemAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or invalid annotation content"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
    },
)
def set_dataset_item_annotations(
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_item_annotations: Annotated[
        SetDatasetItemAnnotations, Body(openapi_examples=SET_DATASET_ITEM_ANNOTATIONS_BODY_EXAMPLES)
    ],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemAnnotations:
    """Set dataset item annotations"""
    try:
        dataset_item = dataset_service.set_dataset_item_annotations(
            project=project,
            dataset_item_id=dataset_item_id,
            annotations=dataset_item_annotations.annotations,
            # Annotations submitted via API are considered user-reviewed, unlike auto-generated predictions
            user_reviewed=True,
        )
        return DatasetItemAnnotations(
            annotations=dataset_item.annotation_data,  # type: ignore[arg-type]
            prediction_model_id=dataset_item.prediction_model_id,
            user_reviewed=dataset_item.user_reviewed,
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AnnotationValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{dataset_item_id}/annotations",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Dataset item found", "model": DatasetItemAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {
            "description": "Dataset item or project not found or dataset item is not annotated"
        },
    },
)
def get_dataset_item_annotations(
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetItemAnnotations:
    """Get the dataset item annotations"""
    try:
        dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
        if dataset_item.annotation_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dataset item has not been annotated yet."
            )
        return DatasetItemAnnotations(
            annotations=dataset_item.annotation_data,
            prediction_model_id=dataset_item.prediction_model_id,
            user_reviewed=dataset_item.user_reviewed,
        )
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
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> None:
    """Delete dataset item annotations"""
    try:
        dataset_service.delete_dataset_item_annotations(project=project, dataset_item_id=dataset_item_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{dataset_item_id}/subset",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Dataset item subset is assigned"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item or project not found"},
        status.HTTP_409_CONFLICT: {"description": "Dataset item already has a subset assigned"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid subset"},
    },
)
def assign_dataset_item_subset(
    project: Annotated[Project, Depends(get_project)],
    dataset_item_id: DatasetItemID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    subset_config: Annotated[DatasetItemAssignSubset, Body()],
) -> DatasetItemView:
    """Assign dataset item subset"""
    try:
        dataset_item = dataset_service.assign_dataset_item_subset(
            project_id=project.id, dataset_item_id=dataset_item_id, subset=subset_config.subset
        )
        return DatasetItemView.model_validate(dataset_item, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except SubsetAlreadyAssignedError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
