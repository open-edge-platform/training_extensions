# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing model architectures"""

from fastapi import APIRouter, status

from app.api.schemas.model_architecture import (
    RECOMMENDED_MODEL_ARCHITECTURES,
    ModelArchitectures,
    ModelArchitectureView,
    TopPicks,
)
from app.models import TaskType
from app.models.model_architecture import ModelArchitecture
from app.supported_models import SupportedModels

router = APIRouter(prefix="/api/model_architectures", tags=["Model Architectures"])


@router.get(
    "",
    response_model=ModelArchitectures,
    responses={
        status.HTTP_200_OK: {"description": "List of available model architectures"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid task type provided"},
    },
)
def get_model_architectures(task: TaskType) -> ModelArchitectures:
    """
    Get all available model architectures, optionally filtered by task type.

    Args:
        task: Task type filter (e.g., 'detection', 'classification', 'instance_segmentation')

    Returns:
        ModelArchitectures containing list of model architectures and recommended top picks.
    """

    model_manifests = SupportedModels.get_model_manifests()
    model_architectures = [
        ModelArchitecture.from_manifest(manifest) for manifest in model_manifests.values() if manifest.task == task
    ]

    top_picks = RECOMMENDED_MODEL_ARCHITECTURES.get(task, None)

    return ModelArchitectures(
        model_architectures=[
            ModelArchitectureView.model_validate(model_architecture, from_attributes=True)
            for model_architecture in model_architectures
        ],
        top_picks=TopPicks.model_validate(top_picks, from_attributes=True),
    )
