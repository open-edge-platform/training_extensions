# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated
from uuid import UUID

import aiofiles
from fastapi import APIRouter, Body, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.api.dependencies import get_data_dir, get_job_dir, get_job_queue, get_project_service
from app.api.validators import JobID
from app.core.jobs.control_plane import CancellationResult, JobQueue
from app.core.jobs.models import JobStatus, TrainingJob, TrainingJobParams
from app.schemas import JobRequest, JobView
from app.schemas.job import JobType
from app.services import ProjectService, ResourceNotFoundError

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
    job_dir: Annotated[Path, Depends(get_job_dir)],
    data_dir: Annotated[Path, Depends(get_data_dir)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> JobView:
    """Create a new job and submit it to the scheduler."""
    try:
        project = project_service.get_project_by_id(job_request.project_id)
        job = None
        match job_request.job_type:
            case JobType.TRAIN:
                job = TrainingJob(
                    project_id=job_request.project_id,
                    log_dir=job_dir,
                    data_dir=data_dir,
                    params=TrainingJobParams(
                        model_architecture_id=job_request.parameters.model_architecture_id,
                        parent_model_revision_id=job_request.parameters.parent_model_revision_id,
                        task=project.task,
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
    List all the jobs.

    Note that job details are not persisted across server restarts;
    in other words, this endpoint only returns jobs submitted during the current server session.
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
    """Get detailed information about a specific job."""
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

    This endpoint allows the client to request the cancellation of a job using its unique job ID.
    The cancellation request is processed asynchronously.
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
    job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]
) -> EventSourceResponse:
    """
    Stream real-time status updates for a specific job.

    This endpoint streams job status updates using Server-Sent Events (SSE).
    It sends periodic updates until the job reaches terminal state.
    """
    if not job_queue.get(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return EventSourceResponse(__gen_job_updates(job_id, job_queue))


@router.get("/{job_id}/logs")
async def stream_job_logs(
    job_id: JobID,
    job_dir: Annotated[Path, Depends(get_job_dir)],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> EventSourceResponse:
    """
    Stream real-time log output for a specific job.

    This endpoint streams job logs using Server-Sent Events (SSE). It reads
    the job's log file and yields new lines as they are written, allowing clients
    to follow the job's progress in real-time. The stream continues until the
    client disconnects or an error occurs.
    """
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status >= JobStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job has already completed; logs are no longer available for streaming",
        )

    log_path = job_dir / job.log_file

    if not log_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")

    return EventSourceResponse(__gen_log_stream(job_id, log_path, job_queue))


async def __gen_job_updates(job_id: UUID, job_queue: JobQueue) -> AsyncGenerator[ServerSentEvent]:
    """Generate job status updates."""
    last = None
    while True:
        j = job_queue.get(job_id)
        if not j:
            break
        snap = JobView.of(j).model_dump_json()
        if snap != last:
            yield ServerSentEvent(data=snap)
            last = snap
        if j.status >= JobStatus.DONE:
            break
        await asyncio.sleep(0.1)


async def __gen_log_stream(job_id: UUID, log_path: Path, job_queue: JobQueue) -> AsyncGenerator[ServerSentEvent]:
    """Asynchronously follow a log file and yield new lines as SSE events."""
    try:
        async with aiofiles.open(log_path) as f:
            async for line in f:
                yield ServerSentEvent(data=line.rstrip("\n"))

            while True:
                j = job_queue.get(job_id)
                if not j:
                    break
                line = await f.readline()
                if not line:
                    await asyncio.sleep(0.3)
                    continue
                yield ServerSentEvent(data=line.rstrip("\n"))
                if j.status >= JobStatus.DONE:
                    break
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield ServerSentEvent(data=f"Error reading log file: {e}")
