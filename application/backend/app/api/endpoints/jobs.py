# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.dependencies import get_model_service, get_project_id
from app.schemas import JobRequest, JobResponse
from app.services import ModelService, ResourceNotFoundError

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job successfully created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def submit_job(
    project_id: Annotated[UUID, Depends(get_project_id)],
    job_request: Annotated[JobRequest, Body()],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> JobResponse:
    """
    Create a new job for the project.

    Args:
        project_id (UUID): The ID of the project.
        job_request (JobRequest): The Job request payload.
        model_service (ModelService): The model service dependency.

    Returns:
        JobResponse: The response containing the job ID.
    """
    try:
        # TODO: Implement actual training logic
        _ = model_service, project_id, job_request  # to avoid unused variable warnings
        return JobResponse(job_id=UUID("94939cbe-e692-4423-b9d3-5f6d93823be3"))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
