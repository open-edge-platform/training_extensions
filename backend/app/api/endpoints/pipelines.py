# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipelines"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.api.dependencies import get_pipeline_id, get_pipeline_service
from app.schemas.metrics import PipelineMetrics
from app.schemas.pipeline import Pipeline, PipelineStatus
from app.services import PipelineService, ResourceAlreadyExistsError, ResourceInUseError, ResourceNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipelines", tags=["Pipelines"])

CREATE_PIPELINE_BODY_DESCRIPTION = """
Configuration for the new pipeline. Requires the IDs of a source, sink, and model to be combined into a pipeline.
The fields 'source_id', 'sink_id', and 'model_id' can be set to `None` to only partially initialize the pipeline. 
"""
CREATE_PIPELINE_BODY_EXAMPLES = {
    "basic": Example(
        summary="Fully configured pipeline",
        description="Configuration for a basic pipeline with source, model, and sink",
        value={
            "name": "Production Pipeline",
            "source_id": "d2cbd8d0-17b8-463e-85a2-4aaed031674d",
            "sink_id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
            "model_id": "b0feaabc-da2b-442e-9b3e-55c11c2c2ff2",
        },
    ),
    "minimal": Example(
        summary="Partially configured pipeline",
        description="Pipeline with only a name (components can be assigned later)",
        value={
            "name": "Experimental Pipeline",
        },
    ),
}

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
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Pipeline successfully created", "model": Pipeline},
        status.HTTP_409_CONFLICT: {"description": "Pipeline already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid request body"},
    },
)
async def create_pipeline(
    pipeline_config: Annotated[
        Pipeline, Body(description=CREATE_PIPELINE_BODY_DESCRIPTION, openapi_examples=CREATE_PIPELINE_BODY_EXAMPLES)
    ],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline:
    """Create and configure a new pipeline"""
    try:
        return pipeline_service.create_pipeline(pipeline_config)
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available pipelines", "model": list[Pipeline]},
    },
)
async def list_pipelines(pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)]) -> list[Pipeline]:
    """List the available pipelines"""
    return pipeline_service.list_pipelines()


@router.get(
    "/{pipeline_id}",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline found", "model": Pipeline},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
async def get_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline:
    """Get info about a given pipeline"""
    try:
        return pipeline_service.get_pipeline_by_id(pipeline_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{pipeline_id}",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline successfully updated", "model": Pipeline},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be updated"},
    },
)
async def update_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_config: Annotated[
        dict,
        Body(
            description=UPDATE_PIPELINE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_PIPELINE_BODY_EXAMPLES,
        ),
    ],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline:
    """Reconfigure an existing pipeline"""
    if "status" in pipeline_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'status' field cannot be changed")
    try:
        return pipeline_service.update_pipeline(pipeline_id, pipeline_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/{pipeline_id}:enable",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline successfully enabled"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be enabled"},
    },
)
async def enable_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """
    Activate a pipeline.
    The pipeline will start processing data from the source, run it through the model, and send results to the sink.
    """
    try:
        pipeline_service.update_pipeline(pipeline_id, {"status": PipelineStatus.RUNNING})
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/{pipeline_id}:disable",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline successfully disabled"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
async def disable_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """Stop a pipeline. The pipeline will become idle, and it won't process any data until re-enabled."""
    try:
        pipeline_service.update_pipeline(pipeline_id, {"status": PipelineStatus.IDLE})
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{pipeline_id}/metrics",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline metrics successfully calculated", "model": PipelineMetrics},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID or duration parameter"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
async def get_pipeline_metrics(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    time_window: int = 60,
) -> PipelineMetrics:
    """
    Calculate model metrics for a pipeline over a specified time window.

    Returns inference latency metrics including average, min, max, 95th percentile,
    and latest latency measurements over the specified duration.
    """
    if time_window <= 0 or time_window > 3600:  # Limit to 1 hour max
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Duration must be between 1 and 3600 seconds"
        )

    try:
        return pipeline_service.get_pipeline_metrics(pipeline_id, time_window)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{pipeline_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Pipeline configuration exported as a ZIP file",
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
async def export_pipeline(
    # pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    # pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    # include_model: bool = False,
) -> FileResponse:
    """Export a pipeline to file"""
    raise NotImplementedError


@router.post(":import", status_code=status.HTTP_204_NO_CONTENT)
async def import_pipeline(
    # zip_file: Annotated[
    #     UploadFile, File(description="ZIP file containing the pipeline configuration and optionally model binaries")
    # ],
    # pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """Import a pipeline from file"""
    raise NotImplementedError


@router.delete(
    "/{pipeline_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Pipeline successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pipeline ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline is currently in running state and cannot be deleted"},
    },
)
async def delete_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """Delete a pipeline. Pipelines must be first disabled (status must be idle) before deletion."""
    try:
        pipeline_service.delete_pipeline_by_id(pipeline_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
