# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_training_configuration_service
from app.api.schemas import TrainingConfigurationView
from app.api.validators import ProjectID
from app.services.model_manifest_service import ManifestNotFoundException
from app.services.training_configuration_service import TrainingConfigurationService

router = APIRouter(prefix="/api/projects/{project_id}/training_configuration", tags=["Training Configuration"])


@router.get("")
def get_training_configuration_by_model_architecture(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: ProjectID,
    model_architecture_id: Annotated[str, Query()],
) -> TrainingConfigurationView:
    """
    Get the full training configuration for a given project and model architecture.

    In other words, this endpoint returns the configuration that would be used for training a new model of the
    specified architecture from scratch.
    """
    try:
        training_config = training_configuration_service.get_by_model_architecture(
            project_id=project_id, model_architecture_id=model_architecture_id
        )
        return TrainingConfigurationView.from_training_configuration(training_config)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ManifestNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("", status_code=status.HTTP_200_OK)
def update_training_configuration_for_model_architecture(
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    project_id: ProjectID,
    model_architecture_id: Annotated[str, Query()],
    training_config_update: dict,
) -> TrainingConfigurationView:
    """
    Update the training configuration, or parts of it, for a given project and model architecture.

    The specific parameters to update are specified in the request body, as a set of key-value pairs.
    Each key identifies the parameter to update within the training configuration (with dot notation,
    e.g. "training.early_stopping.patience"), and the value is the new value to set for that parameter.
    Parameters that are not included in the request body will remain unchanged.
    """
    try:
        training_config = training_configuration_service.get_by_model_architecture(
            project_id=project_id, model_architecture_id=model_architecture_id
        )
        training_config.apply_updates(training_config_update)
        return TrainingConfigurationView.from_training_configuration(training_config)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ManifestNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
