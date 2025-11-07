# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing model architectures"""

from fastapi import APIRouter, status

from app.schemas import ModelArchitectures
from app.schemas.model_architecture import ModelArchitecture
from app.supported_models import SupportedModels

router = APIRouter(prefix="/api/model_architectures", tags=["Model Architectures"])


@router.get(
    "",
    response_model=ModelArchitectures,
    responses={
        status.HTTP_200_OK: {"description": "List of available model architectures"},
    },
)
def get_model_architectures(task: str | None = None) -> ModelArchitectures:
    """
    Get all available model architectures, optionally filtered by task type.

    Args:
        task: Optional task type filter (e.g., 'detection', 'classification', 'instance_segmentation')

    Returns:
        ModelArchitectures containing list of model architectures
    """
    model_manifests = SupportedModels.get_model_manifests()
    model_architectures = [
        ModelArchitecture.from_manifest(manifest)
        for manifest in model_manifests.values()
        if not task or manifest.task.lower() == task.lower()
    ]

    return ModelArchitectures(model_architectures=model_architectures)
