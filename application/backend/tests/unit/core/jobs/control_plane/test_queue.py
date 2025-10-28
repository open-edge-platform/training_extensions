# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from uuid import uuid4

import pytest

from app.core.jobs.control_plane.queue import CancellationResult, JobQueue
from app.core.jobs.models import JobStatus


class TestJobQueue:
    """Test cases for JobQueue functionality."""

    def test_init(self):
        """Test JobQueue initialization."""
        queue = JobQueue()
        assert isinstance(queue._queue, asyncio.Queue)
        assert queue._by_id == {}
        assert queue._order == []
        assert queue._cancellation_events == {}
        assert isinstance(queue._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_submit_job(self, fxt_job):
        """Test job submission to queue."""
        queue = JobQueue()
        job = fxt_job()

        await queue.submit(job)

        assert job.id in queue._by_id
        assert job.id in queue._order
        assert queue._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_submit_multiple_jobs_preserves_order(self, fxt_job):
        """Test that multiple jobs maintain FIFO order."""
        queue = JobQueue()
        jobs = [fxt_job() for _ in range(3)]

        for job in jobs:
            await queue.submit(job)

        assert queue._order == [job.id for job in jobs]

    @pytest.mark.asyncio
    async def test_next_runnable_returns_pending_job(self, fxt_job):
        """Test getting next runnable job from queue."""
        queue = JobQueue()
        job = fxt_job()

        await queue.submit(job)
        next_job = await asyncio.wait_for(queue.next_runnable(), timeout=1.0)

        assert next_job == job
        assert next_job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_next_runnable_skips_cancelled_jobs(self, fxt_job):
        """Test that next_runnable skips cancelled jobs."""
        queue = JobQueue()
        cancelled_job = fxt_job()
        cancelled_job.cancel()
        valid_job = fxt_job()

        await queue.submit(cancelled_job)
        await queue.submit(valid_job)

        next_job = await asyncio.wait_for(queue.next_runnable(), timeout=1.0)
        assert next_job == valid_job

    def test_get_existing_job(self, fxt_job):
        """Test retrieving existing job by ID."""
        queue = JobQueue()
        job = fxt_job()
        queue._by_id[job.id] = job

        result = queue.get(job.id)
        assert result == job

    def test_get_nonexistent_job(self):
        """Test retrieving non-existent job returns None."""
        queue = JobQueue()
        fake_id = uuid4()

        result = queue.get(fake_id)
        assert result is None

    def test_list_all_jobs(self, fxt_job):
        """Test listing all jobs in submission order."""
        queue = JobQueue()
        jobs = [fxt_job() for _ in range(3)]

        for job in jobs:
            queue._by_id[job.id] = job
            queue._order.append(job.id)

        result = queue.list_all()
        assert result == jobs

    def test_list_non_completed_jobs(self, fxt_job):
        """Test listing only non-completed jobs."""
        queue = JobQueue()

        pending_job = fxt_job()
        running_job = fxt_job()
        running_job.start()
        done_job = fxt_job()
        done_job.finish()
        failed_job = fxt_job()
        failed_job.fail("error")
        cancelled_job = fxt_job()
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
    def test_cancel_pending_job(self, initial_status, expected_result, expected_status, fxt_job):
        """Test cancelling a pending job."""
        queue = JobQueue()
        job = fxt_job()
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

    def test_get_cancellation_event_creates_new_event(self, fxt_job):
        """Test that get_cancellation_event creates a new event for unknown job."""
        queue = JobQueue()
        job = fxt_job()
        queue._by_id[job.id] = job

        event = queue.get_cancellation_event(job.id)

        assert isinstance(event, asyncio.Event)
        assert job.id in queue._cancellation_events
        assert queue._cancellation_events[job.id] is event
        assert not event.is_set()

    def test_get_cancellation_event_returns_existing_event(self, fxt_job):
        """Test that get_cancellation_event returns existing event."""
        queue = JobQueue()
        job = fxt_job()
        queue._by_id[job.id] = job

        # Get event first time
        event1 = queue.get_cancellation_event(job.id)
        # Get event second time
        event2 = queue.get_cancellation_event(job.id)

        assert event1 is event2
        assert len(queue._cancellation_events) == 1

    def test_get_cancellation_event_sets_event_if_already_cancelling(self, fxt_job):
        """Test that get_cancellation_event sets event immediately if job is already cancelling."""
        queue = JobQueue()
        job = fxt_job()
        job.status = JobStatus.CANCELLING
        queue._by_id[job.id] = job

        event = queue.get_cancellation_event(job.id)

        assert event.is_set()

    def test_cleanup_cancellation_event_removes_event(self, fxt_job):
        """Test that cleanup_cancellation_event removes the event."""
        queue = JobQueue()
        job = fxt_job()
        queue._by_id[job.id] = job

        # Create event
        queue.get_cancellation_event(job.id)
        assert job.id in queue._cancellation_events

        # Clean up event
        queue.cleanup_cancellation_event(job.id)
        assert job.id not in queue._cancellation_events

    def test_cleanup_cancellation_event_handles_nonexistent_job(self):
        """Test that cleanup_cancellation_event handles non-existent job gracefully."""
        queue = JobQueue()
        fake_id = uuid4()

        # Should not raise exception
        queue.cleanup_cancellation_event(fake_id)
        assert fake_id not in queue._cancellation_events

    def test_cancel_running_job_sets_cancellation_event(self, fxt_job):
        """Test that cancelling a running job sets the cancellation event."""
        queue = JobQueue()
        job = fxt_job()
        job.start()
        queue._by_id[job.id] = job

        # Create cancellation event first
        event = queue.get_cancellation_event(job.id)
        assert not event.is_set()

        # Cancel the job
        queue.cancel(job.id)

        assert job.status == JobStatus.CANCELLING
        assert event.is_set()

    def test_cancel_running_job_without_existing_event(self, fxt_job):
        """Test that cancelling a running job works even without pre-existing event."""
        queue = JobQueue()
        job = fxt_job()
        job.start()
        queue._by_id[job.id] = job

        # Cancel the job without creating event first
        queue.cancel(job.id)

        assert job.status == JobStatus.CANCELLING
        # Event should not be created if it didn't exist
        assert job.id not in queue._cancellation_events
