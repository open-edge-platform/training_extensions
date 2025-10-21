# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.jobs.control_plane import JobController, JobQueue
from app.core.jobs.models import Cancelled, Done, ExecutionEvent, Failed, JobStatus, Progress, Started
from app.core.run import RunnerFactory


class MockRunner:
    """Mock runner for testing purposes."""

    def __init__(self, events=None):
        self.events_to_yield = events or [Started(), Progress("progress", 50.0), Done()]
        self.started = False
        self.stopped = False
        self.stop_called = asyncio.Event()

    def events(self) -> Iterator[ExecutionEvent]:
        """Return iterator of events."""
        return iter(self.events_to_yield)

    def start(self) -> "MockRunner":
        """Mark runner as started."""
        self.started = True
        return self

    async def stop(self) -> None:
        """Mark runner as stopped."""
        self.stopped = True
        self.stop_called.set()


class TestJobController:
    """Test cases for JobController orchestration system."""

    @pytest.fixture
    def fxt_job_queue(self):
        queue = Mock(spec=JobQueue)
        queue.next_runnable = AsyncMock()
        queue.is_cancelling = Mock(return_value=False)
        return queue

    @pytest.fixture
    def fxt_runner_factory(self):
        return Mock(spec=RunnerFactory)

    @pytest.fixture
    def fxt_job_controller(self, fxt_job_queue, fxt_runner_factory):
        return JobController(fxt_job_queue, fxt_runner_factory, max_parallel_jobs=2)

    def test_init(self, fxt_job_queue, fxt_runner_factory):
        """Test JobController initialization."""
        controller = JobController(fxt_job_queue, fxt_runner_factory, max_parallel_jobs=3)

        assert controller._jobs_q == fxt_job_queue
        assert controller._runner_factory == fxt_runner_factory
        assert controller._capacity._sem._value == 3
        assert not controller._running
        assert controller._supervisor_task is None
        assert len(controller._tasks) == 0

    @pytest.mark.asyncio
    async def test_start_creates_supervisor_task(self, fxt_job_controller):
        """Test start method creates supervisor task."""
        await fxt_job_controller.start()

        assert fxt_job_controller._running is True
        assert fxt_job_controller._supervisor_task is not None
        assert fxt_job_controller._supervisor_task.get_name() == "supervisor"

        await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_supervisor_and_tasks(self, fxt_job_controller):
        """Test stop method properly cancels all tasks."""
        await fxt_job_controller.start()

        # Add a mock task
        async def _task():
            await asyncio.sleep(0.01)

        fxt_job_controller._tasks.add(asyncio.create_task(_task()))

        await fxt_job_controller.stop()

        assert fxt_job_controller._running is False
        with pytest.raises(asyncio.CancelledError):
            await fxt_job_controller._supervisor_task
        assert fxt_job_controller._supervisor_task.cancelled()

    @pytest.mark.asyncio
    async def test_supervise_loop_processes_jobs(self, fxt_job_controller, fxt_job_queue, fxt_runner_factory, fxt_job):
        """Test supervisor loop processes jobs from queue."""
        job = fxt_job()
        fxt_job_queue.next_runnable.side_effect = [job, asyncio.CancelledError()]

        mock_runner = MockRunner()
        fxt_runner_factory.for_job.return_value = mock_runner

        await fxt_job_controller.start()

        # Give supervisor time to process
        await asyncio.sleep(0.1)

        await fxt_job_controller.stop()

        fxt_job_queue.next_runnable.assert_called()
        fxt_runner_factory.for_job.assert_called_with(job)

    @pytest.mark.asyncio
    async def test_supervise_loop_handles_exceptions(self, fxt_job_controller, fxt_job_queue):
        """Test supervisor loop handles exceptions gracefully."""
        fxt_job_queue.next_runnable.side_effect = [Exception("Test error"), asyncio.CancelledError()]

        with patch("app.core.jobs.control_plane.controller.logger") as mock_logger:
            await fxt_job_controller.start()
            await asyncio.sleep(0.1)
            await fxt_job_controller.stop()

            mock_logger.exception.assert_called_with("Exception during supervise loop")

    @pytest.mark.asyncio
    async def test_handle_job_events_unknown_event(self, fxt_job_controller, fxt_job):
        """Test handling unknown job events logs warning."""
        job = fxt_job()
        event_queue = asyncio.Queue()

        # Add unknown event type
        class UnknownEvent:
            pass

        await event_queue.put(UnknownEvent())
        await event_queue.put(None)  # Signal end

        with patch("app.core.jobs.control_plane.controller.logger") as mock_logger:
            await fxt_job_controller._handle_job_events(job, event_queue)
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "events,expected_status,expected_progress",
        [
            ([Started(), None], JobStatus.RUNNING, 0.0),
            ([Started(), Progress("Mock", 10.0), Progress("Mock", 20.0), None], JobStatus.RUNNING, 20.0),
            ([Started(), Progress("Mock", 50.0), Done()], JobStatus.DONE, 100.0),
            ([Started(), Progress("Mock", 50.0), Failed("error")], JobStatus.FAILED, 50.0),
            ([Started(), Progress("Mock", 50.0), Cancelled()], JobStatus.CANCELLED, 50.0),
        ],
    )
    async def test_handle_job_events_progress_updates(
        self, events, expected_status, expected_progress, fxt_job_controller, fxt_job
    ):
        """Test progress events update job properly."""
        job = fxt_job()
        event_queue = asyncio.Queue()

        for event in events:
            await event_queue.put(event)

        await fxt_job_controller._handle_job_events(job, event_queue)

        assert job.status == expected_status
        assert job.progress == expected_progress

    @pytest.mark.asyncio
    async def test_setup_job_execution_creates_thread_and_cancel_task(
        self, fxt_job_controller, fxt_runner_factory, fxt_job
    ):
        """Test _setup_job_execution creates proper thread and cancel task."""
        job = fxt_job()
        event_queue = asyncio.Queue()

        mock_runner = MockRunner([Done()])
        fxt_runner_factory.for_job.return_value = mock_runner

        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            cancel_task = fxt_job_controller._setup_job_execution(job, mock_runner, event_queue)

            # Verify thread was created and started
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

            # Verify runner was started
            assert mock_runner.started

            # Clean up
            cancel_task.cancel()
            await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_start_job_creates_task(self, fxt_job_controller, fxt_job):
        """Test _start_job creates and tracks task properly."""
        job = fxt_job()

        with patch.object(fxt_job_controller, "_run_job") as mock_run_job:
            initial_task_count = len(fxt_job_controller._tasks)
            fxt_job_controller._start_job(job)

            # Should have added a task
            assert len(fxt_job_controller._tasks) == initial_task_count + 1

            # Task should call _run_job with the job
            mock_run_job.assert_called_with(job)

            # Clean up the created task
            for task in fxt_job_controller._tasks:
                task.cancel()
            await asyncio.gather(*fxt_job_controller._tasks, return_exceptions=True)
