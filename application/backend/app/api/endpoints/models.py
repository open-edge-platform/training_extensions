# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.dependencies import get_model_id, get_model_service, get_project_id
from app.schemas import Label, Model, TrainingRequest, TrainingResponse
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["Models"])


@router.get(
    "",
    response_model=list[Model],
    responses={
        status.HTTP_200_OK: {"description": "List of available models"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_models(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Get all models in a project."""
    try:
        return model_service.list_models(project_id)
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")


@router.get(
    "/{model_id}",
    response_model=Model,
    responses={
        status.HTTP_200_OK: {"description": "Model found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def get_model(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get a specific model by ID."""
    try:
        return model_service.get_model_by_id(project_id=project_id, model_id=model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{model_id}/labels",
    responses={
        status.HTTP_200_OK: {"description": "Model labels found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def get_model_labels(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Label]:
    """Get labels for a specific model."""
    _ = project_id, model_id, model_service
    raise NotImplementedError("Model labels endpoint is not implemented yet")


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Model configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
        status.HTTP_409_CONFLICT: {"description": "Model is used by at least one pipeline"},
    },
)
def delete_model(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Delete a model from a project."""
    try:
        model_service.delete_model_by_id(project_id=project_id, model_id=model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":train",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Training job successfully created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def train_model(
    project_id: Annotated[UUID, Depends(get_project_id)],
    training_request: Annotated[TrainingRequest, Body()],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> TrainingResponse:
    """
    Start training a new model.

    Creates a new training job for the specified model architecture.
    If parent_model_revision_id is provided, the model will be fine-tuned from that revision.
    If parent_model_revision_id is null, the model will be trained from scratch using base weights.

    Args:
        project_id (UUID): The ID of the project.
        training_request (TrainingRequest): The training request payload.
        model_service (ModelService): The model service dependency.

    Returns:
        TrainingResponse: The response containing the training job ID.
    """
    try:
        # TODO: Implement actual training logic
        _ = model_service, project_id, training_request  # to avoid unused variable warnings
        return TrainingResponse(job_id=UUID("94939cbe-e692-4423-b9d3-5f6d93823be3"))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
