# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.openapi.models import Example

from app.api.dependencies import get_project, get_training_configuration_service
from app.api.schemas import TrainingConfigurationView
from app.api.schemas.project import ProjectView
from app.services.model_manifest_service import ManifestNotFoundException
from app.services.training_configuration_service import TrainingConfigurationService

router = APIRouter(prefix="/api/projects/{project_id}/training_configuration", tags=["Training Configuration"])

UPDATE_TRAINING_CONFIG_EXAMPLES = {
    "reconfigure_subset_split": Example(
        summary="Change the training/validation/testing subset split",
        value={
            "dataset_preparation.subset_split.training": 60,
            "dataset_preparation.subset_split.validation": 25,
            "dataset_preparation.subset_split.testing": 15,
        },
    ),
    "reconfigure_max_epochs": Example(
        summary="Set max number of epochs",
        value={
            "training.max_epochs": 50,
        },
    ),
    "reconfigure_learning_rate": Example(
        summary="Set learning rate",
        value={
            "training.learning_rate": 0.003,
        },
    ),
    "enable_color_jitter_augmentation": Example(
        summary=(
            "Enable color jitter as a data augmentation technique, "
            "customizing the range of brightness adjustment (90% to 110% of original brightness in this example)"
        ),
        value={
            "data_augmentation.color_jitter.enable": True,
            "data_augmentation.color_jitter.brightness": [0.9, 1.1],
        },
    ),
}


@router.get("")
def get_training_configuration_by_model_architecture(
    project: Annotated[ProjectView, Depends(get_project)],
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    model_architecture_id: Annotated[str, Query()],
) -> TrainingConfigurationView:
    """
    Get the full training configuration for a given project and model architecture.

    In other words, this endpoint returns the configuration that would be used for training a new model of the
    specified architecture from scratch.
    """
    try:
        training_config = training_configuration_service.get_by_model_architecture(
            project_id=project.id, model_architecture_id=model_architecture_id
        )
        default_config = TrainingConfigurationService.get_default_by_model_architecture(
            model_architecture_id=model_architecture_id
        )
        return TrainingConfigurationView.from_training_configuration(
            training_config, default_config, task_type=project.task.task_type
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ManifestNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("", status_code=status.HTTP_200_OK)
def update_training_configuration_for_model_architecture(
    project: Annotated[ProjectView, Depends(get_project)],
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
    model_architecture_id: Annotated[str, Query()],
    training_config_update: Annotated[dict, Body(openapi_examples=UPDATE_TRAINING_CONFIG_EXAMPLES)],
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
            project_id=project.id, model_architecture_id=model_architecture_id
        )
        training_config.apply_updates(training_config_update)
        training_configuration_service.update(
            project_id=project.id, model_architecture_id=model_architecture_id, training_configuration=training_config
        )
        default_config = TrainingConfigurationService.get_default_by_model_architecture(
            model_architecture_id=model_architecture_id
        )
        return TrainingConfigurationView.from_training_configuration(
            training_config, default_config, task_type=project.task.task_type
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ManifestNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
