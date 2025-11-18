# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_project_id, get_training_configuration_service
from app.api.serializers.configurable_parameters import ConfigurableParametersConverter
from app.api.serializers.training_configuration import TrainingConfigurationConverter
from app.services import ResourceNotFoundError
from app.services.training_configuration_service import TrainingConfigurationService

router = APIRouter(prefix="/api/projects/{project_id}/training_configuration", tags=["Training Configuration"])


@router.get("")
def get_training_configuration(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_architecture_id: Annotated[str | None, Query()] = None,
    model_revision_id: Annotated[UUID | None, Query()] = None,
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
    training_config_update: dict,
    model_architecture_id: Annotated[str | None, Query()] = None,
) -> dict:
    """
    Update the training configuration for a project.

    - If model_architecture_id is provided, updates configuration for that specific model architecture.
    - If not provided, updates the general task-related configuration.
    """
    try:
        converted_config = ConfigurableParametersConverter.configurable_parameters_from_rest(training_config_update)
        updated_config = training_configuration_service.update_training_configuration(
            project_id=project_id,
            training_config_update=converted_config,
            model_architecture_id=model_architecture_id,
        )
        return TrainingConfigurationConverter().training_configuration_to_rest(updated_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
