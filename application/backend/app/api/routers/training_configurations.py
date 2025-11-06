# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_model_architecture_id,
    get_model_revision_id,
    get_project_id,
    get_training_configuration_service,
)
from app.configuration_tools.training_configuration_converter import TrainingConfigurationConverter
from app.services import ResourceNotFoundError
from app.services.training_configuration_service import TrainingConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects/{project_id}/training_configuration", tags=["Training Configuration"])


@router.get("")
def get_training_configuration(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_architecture_id: Annotated[str, Depends(get_model_architecture_id)],
    model_revision_id: Annotated[UUID, Depends(get_model_revision_id)],
) -> dict:
    """
    Get the training configuration for a project.

    If model_architecture_id is provided, returns configuration for that specific model architecture.
    If model_revision_id is provided, returns configuration for a specific trained model.
    If neither is provided, returns only general task-related configuration.
    Note: model_architecture_id and model_revision_id cannot be used together.
    """
    try:
        training_config = training_configuration_service.get_training_configuration(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            model_revision_id=model_revision_id,
        )
        return TrainingConfigurationConverter().training_configuration_to_rest(training_configuration=training_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("", status_code=status.HTTP_200_OK)
def update_training_configuration(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_architecture_id: Annotated[str, Depends(get_model_architecture_id)],
    training_config_update: dict,
) -> dict:
    """
    Update the training configuration for a project.

    - If model_architecture_id is provided, updates configuration for that specific model architecture.
    - If not provided, updates the general task-related configuration.
    Note: model_architecture_id cannot be used with model_revision_id for updates.
    """
    try:
        updated_config = training_configuration_service.update_training_configuration(
            project_id=project_id,
            training_config_update=training_config_update,
            model_architecture_id=model_architecture_id,
        )
        return TrainingConfigurationConverter().training_configuration_to_rest(updated_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
