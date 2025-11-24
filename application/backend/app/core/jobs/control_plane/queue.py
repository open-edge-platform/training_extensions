# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from enum import StrEnum
from uuid import UUID

from loguru import logger

from app.core.jobs.models import Job, JobStatus


class CancellationResult(StrEnum):
    """
    Outcome of a job cancellation request.

    Values:
        PENDING_CANCELLED: The job was pending and is now marked as cancelled.
        RUNNING_CANCELLING: The job was running and is now marked for cancellation.
        IGNORE_CANCEL: The job was already in a terminal or cancelling state; cancellation is ignored.
        NOT_FOUND: No job with the specified ID was found.
    """

    PENDING_CANCELLED = "pending_cancelled"
    RUNNING_CANCELLING = "running_cancelling"
    IGNORE_CANCEL = "ignore_cancel"
    NOT_FOUND = "not_found"


class JobQueue:
    """Holds all jobs in memory: provides FIFO order and state queries."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._by_id: dict[UUID, Job] = {}
        self._order: list[UUID] = []  # preserve submit order for listing
        self._lock = asyncio.Lock()
        self._cancellation_events: dict[UUID, asyncio.Event] = {}  # Events for cancellation notifications

    async def submit(self, job: Job) -> None:
        """Submit a new job to the queue."""
        async with self._lock:
            self._by_id[job.id] = job
            self._order.append(job.id)
            logger.info("Submitted job with ID: {}", job.id)
            await self._queue.put(job)

    async def next_runnable(self) -> Job:
        """Get the next non-canceled job from the queue (FIFO order)."""
        while True:
            job = await self._queue.get()
            if job.status == JobStatus.CANCELLED:
                logger.info("Skipping cancelled job with ID: {}", job.id)
                continue
            logger.debug("Retrieved job from queue ID: {}, status: {}", job.id, job.status)
            return job

    def get(self, job_id: UUID) -> Job | None:
        """Get a job by its ID."""
        return self._by_id.get(job_id)

    def list_all(self) -> list[Job]:
        """List all jobs."""
        return [self._by_id[jid] for jid in self._order if jid in self._by_id]

    def list_non_completed(self) -> list[Job]:
        """List all non-completed jobs."""
        return [job for job in self._by_id.values() if job.status < JobStatus.DONE]

    def cancel(self, job_id: UUID) -> tuple[Job | None, CancellationResult]:
        """Attempt to cancel a job by its ID."""
        job = self._by_id.get(job_id)
        if not job:
            return None, CancellationResult.NOT_FOUND
        if job.status >= JobStatus.CANCELLING:
            logger.debug("Ignore cancellation for terminal job (status: {}, id: {})", job.status, job.id)
            return None, CancellationResult.IGNORE_CANCEL
        if job.status == JobStatus.PENDING:
            job.cancel()
            logger.info("Cancelled pending job with ID: {}", job_id)
            return job, CancellationResult.PENDING_CANCELLED
        if job.status == JobStatus.RUNNING:
            job.cancelling()
            logger.info("Marked running job for cancellation, ID: {}", job_id)
            # Notify any waiting cancellation monitors
            if job_id in self._cancellation_events:
                self._cancellation_events[job_id].set()
            return job, CancellationResult.RUNNING_CANCELLING

        raise ValueError(f"Unexpected job status: {job.status}")

    def __is_cancelling(self, job_id: UUID) -> bool:
        """Check if a job is marked as being cancelled."""
        job = self._by_id.get(job_id)
        return job is not None and job.status == JobStatus.CANCELLING

    def get_cancellation_event(self, job_id: UUID) -> asyncio.Event:
        """Get or create a cancellation event for the specified job."""
        if job_id not in self._cancellation_events:
            self._cancellation_events[job_id] = asyncio.Event()
            # If the job is already in cancelling state, set the event immediately
            if self.__is_cancelling(job_id):
                self._cancellation_events[job_id].set()
        return self._cancellation_events[job_id]

    def cleanup_cancellation_event(self, job_id: UUID) -> None:
        """Clean up the cancellation event for a job."""
        self._cancellation_events.pop(job_id, None)
