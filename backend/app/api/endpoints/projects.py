# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing projects"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example

from app.api.dependencies import get_project_id, get_project_service
from app.schemas.project import Project
from app.services import ResourceAlreadyExistsError, ResourceInUseError, ResourceNotFoundError
from app.services.project_service import ProjectService

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
            "task": {"type": "classification", "exclusive_labels": True, "labels": [{"name": "cat"}, {"name": "dog"}]},
        },
    ),
    "detection": Example(
        summary="Detection project with exclusive labels",
        description="Configuration for a detection project that have 3 labels: Chardonnay, Sauvignon Blanc and "
        "Cabernet Franc",
        value={
            "name": "grapes",
            "task": {
                "type": "detection",
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
            "task": {"type": "segmentation", "exclusive_labels": True, "labels": [{"name": "car"}, {"name": "person"}]},
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=Project,
    responses={
        status.HTTP_201_CREATED: {"description": "Project successfully created"},
        status.HTTP_409_CONFLICT: {"description": "Project already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid request body"},
    },
)
def create_project(
    project_config: Annotated[
        Project, Body(description=CREATE_PROJECT_BODY_DESCRIPTION, openapi_examples=CREATE_PROJECT_BODY_EXAMPLES)
    ],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Create and configure a new project"""
    try:
        return project_service.create_project(project_config)
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    response_model=list[Project],
    responses={
        status.HTTP_200_OK: {"description": "List of available projects"},
    },
)
def list_projects(project_service: Annotated[ProjectService, Depends(get_project_service)]) -> list[Project]:
    """List the available projects"""
    return project_service.list_projects()


@router.get(
    "/{project_id}",
    response_model=Project,
    responses={
        status.HTTP_200_OK: {"description": "Project found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Get info about a given project"""
    try:
        return project_service.get_project_by_id(project_id)
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
