# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from uuid import uuid4

import pytest

from app.core.jobs.control_plane.queue import CancellationResult, JobQueue
from app.core.jobs.models import Job, JobStatus


class TestJobQueue:
    """Test cases for JobQueue functionality."""

    def test_init(self):
        """Test JobQueue initialization."""
        queue = JobQueue()
        assert isinstance(queue._queue, asyncio.Queue)
        assert queue._by_id == {}
        assert queue._order == []
        assert isinstance(queue._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test job submission to queue."""
        queue = JobQueue()
        job = Job()

        await queue.submit(job)

        assert job.id in queue._by_id
        assert job.id in queue._order
        assert queue._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_submit_multiple_jobs_preserves_order(self):
        """Test that multiple jobs maintain FIFO order."""
        queue = JobQueue()
        jobs = [Job() for _ in range(3)]

        for job in jobs:
            await queue.submit(job)

        assert queue._order == [job.id for job in jobs]

    @pytest.mark.asyncio
    async def test_next_runnable_returns_pending_job(self):
        """Test getting next runnable job from queue."""
        queue = JobQueue()
        job = Job()

        await queue.submit(job)
        next_job = await asyncio.wait_for(queue.next_runnable(), timeout=1.0)

        assert next_job == job
        assert next_job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_next_runnable_skips_cancelled_jobs(self):
        """Test that next_runnable skips cancelled jobs."""
        queue = JobQueue()
        cancelled_job = Job()
        cancelled_job.cancel()
        valid_job = Job()

        await queue.submit(cancelled_job)
        await queue.submit(valid_job)

        next_job = await asyncio.wait_for(queue.next_runnable(), timeout=1.0)
        assert next_job == valid_job

    def test_get_existing_job(self):
        """Test retrieving existing job by ID."""
        queue = JobQueue()
        job = Job()
        queue._by_id[job.id] = job

        result = queue.get(job.id)
        assert result == job

    def test_get_nonexistent_job(self):
        """Test retrieving non-existent job returns None."""
        queue = JobQueue()
        fake_id = uuid4()

        result = queue.get(fake_id)
        assert result is None

    def test_list_all_jobs(self):
        """Test listing all jobs in submission order."""
        queue = JobQueue()
        jobs = [Job() for _ in range(3)]

        for job in jobs:
            queue._by_id[job.id] = job
            queue._order.append(job.id)

        result = queue.list_all()
        assert result == jobs

    def test_list_non_completed_jobs(self):
        """Test listing only non-completed jobs."""
        queue = JobQueue()

        pending_job = Job()
        running_job = Job()
        running_job.start()
        done_job = Job()
        done_job.finish()
        failed_job = Job()
        failed_job.fail("error")
        cancelled_job = Job()
        cancelled_job.cancel()

        jobs = [pending_job, running_job, done_job, failed_job, cancelled_job]
        for job in jobs:
            queue._by_id[job.id] = job

        result = queue.list_non_completed()
        assert len(result) == 2
        assert pending_job in result
        assert running_job in result

    @pytest.mark.parametrize(
        "initial_status, expected_result, expected_status",
        [
            (JobStatus.PENDING, CancellationResult.PENDING_CANCELLED, JobStatus.CANCELLED),
            (JobStatus.RUNNING, CancellationResult.RUNNING_CANCELLING, JobStatus.CANCELLING),
            (JobStatus.CANCELLING, CancellationResult.IGNORE_CANCEL, JobStatus.CANCELLING),
            (JobStatus.CANCELLED, CancellationResult.IGNORE_CANCEL, JobStatus.CANCELLED),
            (JobStatus.DONE, CancellationResult.IGNORE_CANCEL, JobStatus.DONE),
            (JobStatus.FAILED, CancellationResult.IGNORE_CANCEL, JobStatus.FAILED),
        ],
    )
    def test_cancel_pending_job(self, initial_status, expected_result, expected_status):
        """Test cancelling a pending job."""
        queue = JobQueue()
        job = Job()
        job.status = initial_status
        queue._by_id[job.id] = job

        result_job, result = queue.cancel(job.id)

        assert result == expected_result
        if expected_result in (CancellationResult.PENDING_CANCELLED, CancellationResult.RUNNING_CANCELLING):
            assert result_job == job
        else:
            assert result_job is None
        assert job.status == expected_status

    def test_cancel_nonexistent_job(self):
        """Test cancelling a non-existent job."""
        queue = JobQueue()
        fake_id = uuid4()

        result_job, result = queue.cancel(fake_id)

        assert result == CancellationResult.NOT_FOUND
        assert result_job is None

    @pytest.mark.parametrize(
        "initial_status, expected",
        [(status, status == JobStatus.CANCELLING) for status in JobStatus],
    )
    def test_is_cancelling(self, initial_status, expected):
        """Test is_cancelling returns True for cancelling job."""
        queue = JobQueue()
        job = Job()
        job.status = initial_status
        queue._by_id[job.id] = job

        assert queue.is_cancelling(job.id) is expected
