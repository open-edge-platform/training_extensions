# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing projects"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from starlette.responses import FileResponse

from app.api.dependencies import get_data_collector, get_label_service, get_project_id, get_project_service
from app.schemas import Label, PatchLabels, ProjectCreate, ProjectUpdateName, ProjectView
from app.services import (
    LabelService,
    ProjectService,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
)
from app.services.data_collect import DataCollector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Projects"])

CREATE_PROJECT_BODY_DESCRIPTION = """
Configuration for creating a new project. Specify the project name, problem type (classification, detection, 
segmentation), and label definitions.
"""
CREATE_PROJECT_BODY_EXAMPLES = {
    "classification": Example(
        summary="Classification project with exclusive labels",
        description="Configuration for a classification project that have 2 labels: cat and dog",
        value={
            "name": "animals",
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                "labels": [{"name": "cat"}, {"name": "dog"}],
            },
        },
    ),
    "detection": Example(
        summary="Detection project with exclusive labels",
        description="Configuration for a detection project that have 3 labels: Chardonnay, Sauvignon Blanc and "
        "Cabernet Franc",
        value={
            "name": "grapes",
            "task": {
                "task_type": "detection",
                "exclusive_labels": True,
                "labels": [{"name": "Chardonnay"}, {"name": "Sauvignon Blanc"}, {"name": "Cabernet Franc"}],
            },
        },
    ),
    "segmentation": Example(
        summary="Segmentation project with exclusive labels",
        description="Configuration for a segmentation project that have 2 labels: car and person",
        value={
            "name": "traffic",
            "task": {
                "task_type": "segmentation",
                "exclusive_labels": True,
                "labels": [{"name": "car"}, {"name": "person"}],
            },
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectView,
    responses={
        status.HTTP_201_CREATED: {"description": "Project successfully created"},
        status.HTTP_409_CONFLICT: {"description": "Project already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid request body"},
    },
)
def create_project(
    project_config: Annotated[
        ProjectCreate, Body(description=CREATE_PROJECT_BODY_DESCRIPTION, openapi_examples=CREATE_PROJECT_BODY_EXAMPLES)
    ],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Create and configure a new project"""
    try:
        return project_service.create_project(project_config)
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    response_model=list[ProjectView],
    responses={
        status.HTTP_200_OK: {"description": "List of available projects"},
    },
)
def list_projects(project_service: Annotated[ProjectService, Depends(get_project_service)]) -> list[ProjectView]:
    """List the available projects"""
    return project_service.list_projects()


@router.get(
    "/{project_id}",
    response_model=ProjectView,
    responses={
        status.HTTP_200_OK: {"description": "Project found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Get info about a given project"""
    try:
        return project_service.get_project_by_id(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{project_id}",
    response_model=ProjectView,
    responses={
        status.HTTP_200_OK: {"description": "Project name updated successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def rename_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_update_name: Annotated[ProjectUpdateName, Body(description="Updated project name")],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Rename a project"""
    try:
        return project_service.update_project_name(project_id, project_update_name.name)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Project successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
        status.HTTP_409_CONFLICT: {"description": "Project has a running pipeline and cannot be deleted"},
    },
)
def delete_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> None:
    """Delete a project. Project with a running pipeline cannot be deleted."""
    try:
        project_service.delete_project_by_id(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.patch(
    "/{project_id}/labels",
    response_model=list[Label],
    responses={
        status.HTTP_200_OK: {"description": "Labels updated successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or label not found"},
        status.HTTP_409_CONFLICT: {"description": "Label(s) already exists"},
    },
)
def update_labels(
    project_id: Annotated[UUID, Depends(get_project_id)],
    labels: Annotated[
        PatchLabels,
        Body(
            description="Labels to add, remove or edit",
        ),
    ],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    label_service: Annotated[LabelService, Depends(get_label_service)],
) -> list[Label]:
    """Update labels for a given project"""
    try:
        project = project_service.get_project_by_id(project_id)
        label_ids = {label.id for label in project.task.labels}
        labels_to_add = [label_to_add.to_label() for label_to_add in labels.labels_to_add]
        labels_to_edit = []
        label_ids_to_remove = []
        if labels.labels_to_remove:
            label_ids_to_remove = [label.id for label in labels.labels_to_remove]
            if not all(label_id in label_ids for label_id in label_ids_to_remove):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="One or more labels to remove do not exist in the project",
                )
        if labels.labels_to_edit:
            ids_to_edit = [label.id for label in labels.labels_to_edit]
            if not all(label_id in label_ids for label_id in ids_to_edit):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="One or more labels to edit do not exist in the project",
                )
            labels_to_edit = [label_to_edit.to_label() for label_to_edit in labels.labels_to_edit]
        return label_service.update_labels_in_project(project_id, labels_to_add, labels_to_edit, label_ids_to_remove)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "/{project_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Project thumbnail found"},
        status.HTTP_204_NO_CONTENT: {"description": "No thumbnail available"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_project_thumbnail(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> FileResponse:
    """Get the project's thumbnail image"""
    try:
        thumbnail_path = project_service.get_project_thumbnail_path(project_id)
        if thumbnail_path:
            return FileResponse(path=thumbnail_path)
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{project_id}/pipeline:capture",
    responses={
        status.HTTP_200_OK: {"description": "Successfully marked next pipeline frame to be collected"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def capture_next_pipeline_frame(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    data_collector: Annotated[DataCollector, Depends(get_data_collector)],
) -> None:
    """Marks next pipeline frame to be collected"""
    try:
        project_service.get_project_by_id(project_id)
        data_collector.collect_next_frame()
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
