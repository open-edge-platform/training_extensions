# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.openapi.models import Example

from app.api.dependencies import get_model_id, get_model_service
from app.schemas import Model
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError

router = APIRouter(prefix="/api/models", tags=["Models"])


UPDATE_MODEL_BODY_EXAMPLES = {
    "rename_model": Example(
        summary="Rename model",
        description="Change the name of the model",
        value={
            "name": "New Model Name",
        },
    )
}


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available models", "model": list[Model]},
    },
)
async def list_models(model_service: Annotated[ModelService, Depends(get_model_service)]) -> list[Model]:
    """Get information about available models"""
    return model_service.list_models()


@router.get(
    "/{model_id}",
    responses={
        status.HTTP_200_OK: {"description": "Model found", "model": Model},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Model not found"},
    },
)
async def get_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get information about a specific model"""
    try:
        return model_service.get_model_by_id(model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Model configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Model not found"},
        status.HTTP_409_CONFLICT: {"description": "Model is used by at least one pipeline"},
    },
)
async def delete_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Delete a model"""
    try:
        model_service.delete_model_by_id(model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
