# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from starlette.responses import StreamingResponse

from app.api.dependencies import get_job_queue, get_project_service
from app.api.validators import JobID
from app.core.jobs.control_plane import CancellationResult, JobQueue
from app.core.jobs.models import JobStatus
from app.schemas import JobRequest, JobView
from app.schemas.job import JobType
from app.services import ProjectService, ResourceNotFoundError
from app.services.training.models import TrainingJob, TrainingParams

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job successfully created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Unknown job type or invalid parameters"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
async def submit_job(
    job_request: Annotated[JobRequest, Body()],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> JobView:
    """
    Create a new job for the project.

    Args:
        job_request (JobRequest): The Job request payload.
        job_queue (JobQueue): The job queue instance responsible for managing job submissions and tracking job statuses.
        project_service (ProjectService): The service to interact with project data.

    Returns:
        JobView: The response containing the job ID.
    """
    try:
        project = project_service.get_project_by_id(job_request.project_id)
        job = None
        match job_request.job_type:
            case JobType.TRAIN:
                job = TrainingJob(
                    project_id=job_request.project_id,
                    params=TrainingParams(
                        model_architecture_id=job_request.parameters.model_architecture_id,
                        parent_model_revision_id=job_request.parameters.parent_model_revision_id,
                        task_type=project.task.task_type,
                    ),
                )
            case _:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown job type")
        await job_queue.submit(job)
        return JobView.of(job)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "",
    response_model=list[JobView],
    responses={
        status.HTTP_200_OK: {"description": "List all jobs"},
    },
)
async def list_jobs(job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> list[JobView]:
    """
    Retrieve a list of all jobs.

    This endpoint returns a list of all jobs that have been registered
    during the active server session.

    Args:
        job_queue (JobQueue): The job queue instance responsible for managing job submissions and tracking job statuses.

    Returns:
        list[JobView]: A list of job responses.
    """
    return [JobView.of(job) for job in job_queue.list_all()]


@router.get(
    "/{job_id}",
    response_model=JobView,
    responses={
        status.HTTP_200_OK: {"description": "Job found"},
        status.HTTP_404_NOT_FOUND: {"description": "Job with ID not found"},
    },
)
async def get_job(job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> JobView:
    """
    Retrieve details of a specific job.

    This endpoint fetches the details of a job using its unique job ID.

    Args:
        job_id (JobID): The unique identifier of the job.
        job_queue (JobQueue): The job queue instance responsible for managing job submissions and tracking job statuses.

    Returns:
        JobView: The response containing the job details.
    """
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobView.of(job)


@router.post(
    "/{job_id}:cancel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobView,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job cancel successfully requested"},
        status.HTTP_404_NOT_FOUND: {"description": "Job with ID not found"},
        status.HTTP_409_CONFLICT: {"description": "Job cannot be canceled in its current state"},
    },
)
async def cancel_job(job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> JobView:
    """
    Request cancellation of a specific job.

    This endpoint allows the client to request the cancellation of a job
    using its unique job ID. The cancellation request is processed asynchronously.

    Args:
        job_id (JobID): The unique identifier of the job to be canceled.
        job_queue (JobQueue): The job queue instance responsible for managing job submissions and tracking job statuses.

    Returns:
        JobView: The response containing the job ID of the canceled job.

    Raises:
        HTTPException: If the job is not found (404) or the cancellation fails (409).
    """
    try:
        job, result = job_queue.cancel(job_id)
        match result:
            case CancellationResult.NOT_FOUND:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
            case CancellationResult.IGNORE_CANCEL:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job already completed or cancelled")
            case CancellationResult.PENDING_CANCELLED | CancellationResult.RUNNING_CANCELLING:
                if not job:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Job not found")
                return JobView.of(job)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unable to cancel job")


@router.get("/{job_id}/status")
async def stream_job_status(
    job_id: JobID, request: Request, job_queue: Annotated[JobQueue, Depends(get_job_queue)]
) -> StreamingResponse:
    """
    Stream real-time status updates for a specific job.

    This endpoint streams job status updates using Server-Sent Events (SSE).
    It sends periodic updates until the client disconnects or the job reaches
    terminal state.

    Args:
        job_id (JobID): The unique identifier of the job.
        request (Request): The HTTP request object to monitor client connection status.
        job_queue (JobQueue): The job queue instance responsible for managing job submissions and tracking job statuses.

    Returns:
        StreamingResponse: A streaming response with job status updates.
    """
    if not job_queue.get(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    async def gen_job_updates() -> AsyncGenerator[str]:
        """Generate job status updates."""
        last = None
        while True:
            if await request.is_disconnected():
                break
            j = job_queue.get(job_id)
            if not j:
                break
            snap = JobView.of(j).model_dump_json()
            if snap != last:
                yield f"{snap}\n"
                last = snap
            if j.status >= JobStatus.DONE:
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(
        gen_job_updates(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        },
    )
