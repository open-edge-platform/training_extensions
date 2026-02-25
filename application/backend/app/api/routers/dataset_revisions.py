# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.openapi.models import Example
from starlette.responses import StreamingResponse

from app.api.dependencies import get_dataset_revision, get_dataset_revision_service, get_project
from app.api.io_utils import write_file_to_response, write_image_to_response
from app.api.schemas.dataset_item import DatasetRevisionItemsWithPagination, DatasetRevisionItemView
from app.api.schemas.dataset_revision import DatasetRevisionView
from app.api.validators import DatasetItemID, DatasetRevisionID
from app.core.models import Pagination
from app.models import DatasetItemSubset, Project
from app.models.dataset_revision import DatasetRevision
from app.services import DatasetRevisionService

router = APIRouter(
    prefix="/api/projects/{project_id}/dataset_revisions",
    tags=["Dataset Revisions"],
)

DEFAULT_DATASET_ITEMS_NUMBER_RETURNED = 10
MAX_DATASET_ITEMS_NUMBER_RETURNED = 100

UPDATE_DATASET_REVISION_BODY_DESCRIPTION = """
Update name of a dataset revision.
"""
UPDATE_DATASET_REVISION_BODY_EXAMPLES = {
    "name": Example(
        summary="Update dataset revision name",
        description="Change the name of a dataset revision",
        value={
            "name": "new_dataset_revision_name",
        },
    ),
}


@router.get(
    "",
    response_model=list[DatasetRevisionView],
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset revisions"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_dataset_revisions(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> list[DatasetRevisionView]:
    """List the dataset revisions in a project."""
    return [
        DatasetRevisionView.model_validate(dataset_revision.model_dump(), from_attributes=True)
        for dataset_revision in dataset_revision_service.list_dataset_revisions(project_id=project.id)
    ]


@router.get(
    "/{dataset_revision_id}",
    response_model=DatasetRevisionView,
    responses={
        status.HTTP_200_OK: {"description": "Dataset revision found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or dataset revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset revision not found"},
    },
)
def get_dataset_revision_details(
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
) -> DatasetRevisionView:
    """Get information about a specific dataset revision."""
    return DatasetRevisionView.model_validate(dataset_revision.model_dump(), from_attributes=True)


@router.patch(
    "/{dataset_revision_id}",
    response_model=DatasetRevisionView,
    responses={
        status.HTTP_200_OK: {"description": "Dataset revision successfully renamed"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or dataset revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset revision not found"},
    },
)
def rename_dataset_revision(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_revision_metadata: Annotated[
        dict,
        Body(
            description=UPDATE_DATASET_REVISION_BODY_DESCRIPTION,
            openapi_examples=UPDATE_DATASET_REVISION_BODY_EXAMPLES,
        ),
    ],
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> DatasetRevisionView:
    """Rename a dataset revision."""
    try:
        new_name = dataset_revision_metadata["name"]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'name' field in request body",
        )
    dataset_revision = dataset_revision_service.rename_dataset_revision(
        project_id=project.id,
        dataset_revision=dataset_revision,
        new_name=new_name,
    )
    return DatasetRevisionView.model_validate(dataset_revision.model_dump(), from_attributes=True)


@router.get(
    "/{dataset_revision_id}/items",
    responses={
        status.HTTP_200_OK: {
            "description": "List of dataset items in the revision",
            "model": DatasetRevisionItemsWithPagination,
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
) -> DatasetRevisionItemsWithPagination:
    """List the items in a dataset revision. This endpoint supports pagination."""
    dataset_revision_items, total_count = dataset_revision_service.list_dataset_revision_items(
        project_id=project.id,
        dataset_revision=dataset_revision,
        limit=limit,
        offset=offset,
        subset=subset,
    )

    return DatasetRevisionItemsWithPagination(
        items=[DatasetRevisionItemView.model_validate(item, from_attributes=True) for item in dataset_revision_items],
        pagination=Pagination(
            total=total_count,
            limit=limit,
            offset=offset,
            count=len(dataset_revision_items),
        ),
    )


@router.get(
    "/{dataset_revision_id}/items/{dataset_item_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset revision item found", "model": DatasetRevisionItemView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset item, revision, or project not found"},
    },
)
def get_dataset_revision_item(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> DatasetRevisionItemView:
    """Get information about a specific item in the dataset revision"""
    dataset_revision_item = dataset_revision_service.get_dataset_revision_item(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    )
    return DatasetRevisionItemView.model_validate(dataset_revision_item, from_attributes=True)


@router.get(
    "/{dataset_revision_id}/items/{dataset_item_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset revision, item or project not found"},
    },
)
def get_dataset_revision_item_binary(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> StreamingResponse:
    """Get the image data of an item in the dataset revision"""
    binary_path = dataset_revision_service.get_dataset_revision_item(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    ).image_path
    return write_file_to_response(path=binary_path, filename=f"{dataset_item_id}.jpeg")


@router.get(
    "/{dataset_revision_id}/items/{dataset_item_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Dataset item thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid dataset item ID or revision ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Dataset revision, item or project not found"},
    },
)
def get_dataset_revision_item_thumbnail(
    project: Annotated[Project, Depends(get_project)],
    dataset_revision: Annotated[DatasetRevision, Depends(get_dataset_revision)],
    dataset_item_id: DatasetItemID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> StreamingResponse:
    """Get the thumbnail of an item in the dataset revision"""
    thumbnail = dataset_revision_service.get_dataset_revision_item_thumbnail(
        project_id=project.id,
        dataset_revision=dataset_revision,
        item_id=str(dataset_item_id),
    )
    return write_image_to_response(
        image=thumbnail, filename=f"{dataset_item_id}_thumb.jpeg", cache_control="public, max-age=31536000"
    )


@router.delete(
    "/{dataset_revision_id}",
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
    """
    Delete the files relative to a given dataset revision to free up disk space.

    Specifically, this endpoint deletes all items and their associated binary data (images, annotations, etc.)
    from the dataset revision, but it does NOT delete the revision metadata (such as name, creation date, etc.).
    After invoking this endpoint, the dataset revision will still exist in the system, but it will no longer be possible
    to view the items contained within it.

    Beware that this operation is irreversible; once the files are deleted, they cannot be recovered.
    Also note that the deletion of dataset revision items does NOT have any impact on the project main dataset,
    i.e. the dataset items returned by the '/dataset/items' endpoints.

    The only purpose of this endpoint is to free up storage space by deleting files no longer deemed necessary,
    namely snapshots of datasets used during training (dataset revisions).
    """
    dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=dataset_revision_id)
