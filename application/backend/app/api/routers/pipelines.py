# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipelines"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from pydantic import ValidationError

from app.api.dependencies import get_pipeline_metrics_service, get_pipeline_service, get_system_service
from app.api.schemas import PipelineView
from app.api.validators import ProjectID
from app.models import DataCollectionPolicyAdapter, PipelineStatus
from app.schemas.metrics import PipelineMetrics
from app.services import PipelineMetricsService, PipelineService, ResourceNotFoundError, SystemService

router = APIRouter(prefix="/api/projects/{project_id}/pipeline", tags=["Pipelines"])

UPDATE_PIPELINE_BODY_DESCRIPTION = """
Partial pipeline configuration update. May contain any subset of fields including 'name', 'source_id', 
'sink_id', or 'model_id'. Fields not included in the request will remain unchanged.
"""
UPDATE_PIPELINE_BODY_EXAMPLES = {
    "switch_model": Example(
        summary="Switch active model",
        description="Change the active model for the pipeline",
        value={
            "model_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
        },
    ),
    "reconfigure": Example(
        summary="Reconfigure pipeline",
        description="Change the name, source and sink of the pipeline",
        value={
            "name": "Updated Production Pipeline",
            "source_id": "e3cbd8d0-17b8-463e-85a2-4aaed031674e",
            "sink_id": "c6787c06-964b-4097-8eca-238b8cf79fc9",
        },
    ),
    "enable_data_collection_policies": Example(
        summary="Enable data collection policies",
        description="Change data collection policies of the pipeline to fixed rate",
        value={
            "data_collection_policies": [
                {
                    "type": "fixed_rate",
                    "enabled": "true",
                    "rate": 0.1,
                }
            ]
        },
    ),
    "clean_data_collection_policies": Example(
        summary="Clean data collection policies",
        description="Remove all data collection policies of the pipeline",
        value={"data_collection_policies": []},
    ),
    "change_device": Example(
        summary="Change inference device",
        description="Change the device used for model inference (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')",
        value={"device": "xpu"},
    ),
}


@router.get(
    "",
    response_model=PipelineView,
    responses={
        status.HTTP_200_OK: {"description": "Pipeline found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
def get_pipeline(
    project_id: ProjectID,
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> PipelineView:
    """Get info about a given pipeline"""
    try:
        pipeline = pipeline_service.get_pipeline_by_id(project_id)
        return PipelineView.model_validate(pipeline, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "",
    response_model=PipelineView,
    responses={
        status.HTTP_200_OK: {"description": "Pipeline successfully reconfigured"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be reconfigured"},
    },
)
def update_pipeline(
    project_id: ProjectID,
    pipeline_config: Annotated[
        dict,
        Body(
            description=UPDATE_PIPELINE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_PIPELINE_BODY_EXAMPLES,
        ),
    ],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> PipelineView:
    """Reconfigure an existing pipeline"""
    if "status" in pipeline_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'status' field cannot be changed")

    if "device" in pipeline_config:
        device_str = pipeline_config["device"]
        if not system_service.validate_device(device_str):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=f"Device '{device_str}' is not available on this system"
            )

    try:
        if "data_collection_policies" in pipeline_config:
            pipeline_config["data_collection_policies"] = [
                DataCollectionPolicyAdapter.validate_python(policy)
                for policy in pipeline_config["data_collection_policies"]
            ]
        updated = pipeline_service.update_pipeline(project_id, pipeline_config)
        return PipelineView.model_validate(updated, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":enable",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline successfully enabled"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be enabled"},
    },
)
def enable_pipeline(
    project_id: ProjectID,
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """
    Activate a pipeline.
    The pipeline will start processing data from the source, run it through the model, and send results to the sink.
    """
    try:
        pipeline_service.update_pipeline(project_id, {"status": PipelineStatus.RUNNING})
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":disable",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline successfully disabled"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
def disable_pipeline(
    project_id: ProjectID,
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """Stop a pipeline. The pipeline will become idle, and it won't process any data until re-enabled."""
    try:
        pipeline_service.update_pipeline(project_id, {"status": PipelineStatus.IDLE})
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/metrics",
    response_model=PipelineMetrics,
    responses={
        status.HTTP_200_OK: {"description": "Pipeline metrics successfully calculated"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or duration parameter"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
def get_project_metrics(
    project_id: ProjectID,
    pipeline_metrics_service: Annotated[PipelineMetricsService, Depends(get_pipeline_metrics_service)],
    time_window: int = 60,
) -> PipelineMetrics:
    """
    Calculate model metrics for a pipeline over a specified time window.

    Returns inference latency and throughput metrics including average, min, max, 95th percentile,
    and latest latency measurements, plus throughput data over the specified duration.
    """
    if time_window <= 0 or time_window > 3600:  # Limit to 1 hour max
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Duration must be between 1 and 3600 seconds"
        )

    try:
        return pipeline_metrics_service.get_pipeline_metrics(project_id, time_window)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
