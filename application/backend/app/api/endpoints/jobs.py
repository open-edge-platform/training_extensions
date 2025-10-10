# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Request, status
from starlette.responses import StreamingResponse

from app.api.validators import JobID
from app.schemas import JobRequest, JobResponse
from app.services import ResourceNotFoundError

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job successfully created"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
async def submit_job(
    job_request: Annotated[JobRequest, Body()],
) -> JobResponse:
    """
    Create a new job for the project.

    Args:
        job_request (JobRequest): The Job request payload.

    Returns:
        JobResponse: The response containing the job ID.
    """
    try:
        # TODO: Request job scheduling using Jobs Control Plane (TBD)
        _ = job_request
        return JobResponse(job_id=uuid4())
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "",
    response_model=list[JobResponse],
    responses={
        status.HTTP_200_OK: {"description": "List all jobs"},
    },
)
async def list_jobs() -> list[JobResponse]:
    """
    Retrieve a list of all jobs.

    This endpoint returns a list of all jobs that have been registered
    during the active server session.

    Returns:
        list[JobResponse]: A list of job responses.
    """
    return [JobResponse(job_id=uuid4())]


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={
        status.HTTP_200_OK: {"description": "Job found"},
    },
)
async def get_job(job_id: JobID) -> JobResponse:
    """
    Retrieve details of a specific job.

    This endpoint fetches the details of a job using its unique job ID.

    Args:
        job_id (JobID): The unique identifier of the job.

    Returns:
        JobResponse: The response containing the job details.
    """
    return JobResponse(job_id=job_id)


@router.post(
    "/{job_id}:cancel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job cancel successfully requested"},
        status.HTTP_404_NOT_FOUND: {"description": "Job with ID not found"},
        status.HTTP_409_CONFLICT: {"description": "Job cannot be canceled in its current state"},
    },
)
async def cancel_job(job_id: JobID) -> JobResponse:
    """
    Request cancellation of a specific job.

    This endpoint allows the client to request the cancellation of a job
    using its unique job ID. The cancellation request is processed asynchronously.

    Args:
        job_id (JobID): The unique identifier of the job to be canceled.

    Returns:
        JobResponse: The response containing the job ID of the canceled job.

    Raises:
        HTTPException: If the job is not found (404) or the cancellation fails (409).
    """
    return JobResponse(job_id=job_id)


@router.get("/{job_id}/status")
async def stream_job_status(job_id: JobID, request: Request) -> StreamingResponse:
    """
    Stream real-time status updates for a specific job.

    This endpoint streams job status updates using Server-Sent Events (SSE).
    It sends periodic updates until the client disconnects or the job reaches
    terminal state.

    Args:
        job_id (JobID): The unique identifier of the job.
        request (Request): The HTTP request object to monitor client
        connection status.

    Returns:
        StreamingResponse: A streaming response with job status updates.
    """

    async def gen_job_updates() -> AsyncGenerator[str]:
        """Generate job status updates."""
        for _ in range(20):
            if await request.is_disconnected():
                break
            yield f"Hey there from {job_id}\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen_job_updates(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        },
    )
