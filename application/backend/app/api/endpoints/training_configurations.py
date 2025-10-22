# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_training_configuration_service
from app.configuration_tools.training_configuration_converter import convert_training_configuration_to_rest
from app.services import ResourceNotFoundError
from app.services.training_configuration_service import TrainingConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects/{project_id}/training_configuration", tags=["Training Configuration"])


@router.get("")
def get_training_configuration(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: UUID,
    model_architecture_id: str | None = None,
    model_revision_id: UUID | None = None,
) -> dict:
    """
    Get the training configuration for a project.

    If model_architecture_id is provided, returns configuration for that specific model architecture.
    If model_revision_id is provided, returns configuration for a specific trained model.
    If neither is provided, returns only general task-related configuration.
    Note: model_architecture_id and model_revision_id cannot be used together.

    Args:
        training_configuration_service (TrainingConfigurationService): The training configuration service.
        project_id (UUID): The unique identifier of the project.
        model_architecture_id (Optional[str]): The model architecture ID for specific configuration retrieval.
        model_revision_id (Optional[UUID]): The model revision ID for specific configuration retrieval.

    Returns:
        TrainingConfiguration: The training configuration details.
    """
    try:
        training_configuration = training_configuration_service.get_training_configuration(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            model_revision_id=model_revision_id,
        )
        return convert_training_configuration_to_rest(config=training_configuration)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("", status_code=status.HTTP_204_NO_CONTENT)
def update_training_configuration(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: UUID,
    training_config_update: dict,
    model_architecture_id: str | None = None,
) -> None:
    """
    Update the training configuration for a project.

    - If model_architecture_id is provided, updates configuration for that specific model architecture.
    - If not provided, updates the general task-related configuration.
    Note: model_architecture_id cannot be used with model_revision_id for updates.

    Request body should contain elements of the configuration hierarchy to update:
    ```json
    {
      "dataset_augmentation_parameters": {...},
      "training": {...},
      "evaluation": {...}
    }
    ```

    Args:
        training_configuration_service (TrainingConfigurationService): The training configuration service.
        project_id (UUID): The unique identifier of the project.
        training_config_update (dict): The configuration updates to apply.
        model_architecture_id (Optional[str]): The model architecture ID for specific configuration update.
    """
    try:
        training_configuration_service.update_training_configuration(
            project_id=project_id,
            training_config_update=training_config_update,
            model_architecture_id=model_architecture_id,
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
