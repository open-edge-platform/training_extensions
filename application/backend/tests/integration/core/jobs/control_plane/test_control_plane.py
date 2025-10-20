# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock

import pytest

from app.core.jobs.control_plane import CancellationResult, JobController, JobQueue
from app.core.jobs.exec import ThreadRunnerFactory
from app.core.jobs.models import Job, JobStatus
from app.core.run import ExecutionContext, RunnableFactory

from .mock_runnable import MockRunnable, RunnableBehaviour


class TestJobControlPlaneIntegration:
    """Integration tests for the job control plane with thread runner architecture."""

    @pytest.fixture
    def fxt_runnable_factory(self):
        factory = Mock(spec=RunnableFactory)
        factory.return_value = MockRunnable()
        return factory

    @pytest.fixture
    def fxt_job_queue(self):
        return JobQueue()

    @pytest.fixture
    def fxt_runner_factory(self, fxt_runnable_factory):
        return ThreadRunnerFactory(fxt_runnable_factory)

    @pytest.fixture
    def fxt_job_controller(self, fxt_job_queue, fxt_runner_factory):
        return JobController(fxt_job_queue, fxt_runner_factory, max_parallel_jobs=2)

    @pytest.mark.asyncio
    async def test_successful_job_execution(self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job):
        """Test complete job lifecycle from submission to successful completion."""
        # Set up successful runnable
        fxt_runnable_factory.return_value = MockRunnable(behavior=RunnableBehaviour.SUCCESS)

        # Create and submit job
        job = fxt_job()
        await fxt_job_queue.submit(job)

        # Start controller
        await fxt_job_controller.start()

        try:
            # Wait for job to complete with proper timeout
            await self._wait_for_job_status(job, JobStatus.DONE, timeout=5.0)

            # Verify job completed successfully
            assert job.status == JobStatus.DONE
            assert job.progress == 100.0
            assert job.started_at is not None
            assert job.error is None

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_job_failure_handling(self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job):
        """Test job failure is properly handled and propagated."""
        # Set up failing runnable
        fxt_runnable_factory.return_value = MockRunnable(behavior=RunnableBehaviour.FAILURE)

        job = fxt_job()
        await fxt_job_queue.submit(job)

        await fxt_job_controller.start()

        try:
            # Wait for job to fail
            await self._wait_for_job_status(job, JobStatus.FAILED, timeout=5.0)

            # Verify job failed with error details
            assert job.status == JobStatus.FAILED
            assert "Mock failure" in job.error
            assert job.started_at is not None

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_job_cancellation_during_execution(
        self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job
    ):
        """Test job cancellation during active execution."""
        # Use slow runnable for cancellation testing
        fxt_runnable_factory.return_value = MockRunnable(behavior=RunnableBehaviour.SLOW, execution_time=0.05)

        job = fxt_job()
        await fxt_job_queue.submit(job)

        await fxt_job_controller.start()

        try:
            # Wait for job to start running
            await self._wait_for_job_status(job, JobStatus.RUNNING, timeout=2.0)

            # Cancel running job
            result_job, result = fxt_job_queue.cancel(job.id)
            assert result == CancellationResult.RUNNING_CANCELLING

            # Wait for cancellation to complete
            await self._wait_for_job_status(job, JobStatus.CANCELLED, timeout=2.0)

            # Job should be cancelled
            assert job.status == JobStatus.CANCELLED

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_multiple_jobs_concurrent_execution(
        self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job
    ):
        """Test multiple jobs can execute concurrently within capacity limits."""
        # Create jobs with fast execution
        jobs = []
        for i in range(3):
            job = fxt_job()
            jobs.append(job)
            await fxt_job_queue.submit(job)

        # Mock factory to return fast success runnables
        fxt_runnable_factory.return_value = MockRunnable(behavior=RunnableBehaviour.INSTANT)

        await fxt_job_controller.start()

        try:
            # Wait for all jobs to complete
            for job in jobs:
                await self._wait_for_job_status(job, JobStatus.DONE, timeout=3.0)

            # All jobs should complete successfully
            for job in jobs:
                assert job.status == JobStatus.DONE
                assert job.progress == 100.0

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_capacity_management_limits_concurrency(self, fxt_runnable_factory, fxt_job):
        """Test that capacity management properly limits concurrent execution."""
        job_queue = JobQueue()
        runner_factory = ThreadRunnerFactory(fxt_runnable_factory)

        # Create controller with capacity of 1 for clear sequential testing
        job_controller = JobController(job_queue, runner_factory, max_parallel_jobs=1)

        # Track concurrent execution
        concurrent_count = 0
        max_concurrent = 0

        class ConcurrencyTrackingRunnable(MockRunnable):
            def run(self, ctx: ExecutionContext):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

                try:
                    # Simulate some work
                    for progress in [50.0, 100.0]:
                        ctx.report_progress(progress=progress)
                        ctx.heartbeat()
                        # Small delay to ensure overlap if running concurrently
                        import time

                        time.sleep(0.05)
                finally:
                    concurrent_count -= 1

        # Create multiple jobs
        jobs = []
        for i in range(3):
            job = fxt_job()
            jobs.append(job)
            await job_queue.submit(job)

        # Mock factory returns concurrency tracking runnable
        fxt_runnable_factory.return_value = ConcurrencyTrackingRunnable()

        await job_controller.start()

        try:
            # Wait for all jobs to complete
            for job in jobs:
                await self._wait_for_job_status(job, JobStatus.DONE, timeout=5.0)

            # Verify capacity was respected (max 1 concurrent)
            assert max_concurrent == 1, f"Expected max 1 concurrent job, got {max_concurrent}"

            # All jobs should complete
            assert all(job.status == JobStatus.DONE for job in jobs)

        finally:
            await job_controller.stop()

    @pytest.mark.asyncio
    async def test_progress_updates_propagation(self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job):
        """Test that progress updates are properly propagated to job state."""
        fxt_runnable_factory.return_value = MockRunnable(
            behavior=RunnableBehaviour.SUCCESS, progress_steps=[30.0, 70.0]
        )

        job = fxt_job()
        await fxt_job_queue.submit(job)

        await fxt_job_controller.start()

        try:
            # Wait for job to complete
            await self._wait_for_job_status(job, JobStatus.PENDING, timeout=3.0)
            assert job.started_at is None

            # Should transition to running
            await self._wait_for_job_status(job, JobStatus.RUNNING, timeout=3.0)
            assert job.progress == 30.0
            assert job.started_at is not None

            # Should complete
            await self._wait_for_job_status(job, JobStatus.DONE, timeout=3.0)
            assert job.updated_at > job.started_at

            # Job should be complete with final progress
            assert job.status == JobStatus.DONE
            assert job.progress == 100.0

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_multiple_job_types_mixed_execution(
        self, fxt_job_queue, fxt_job_controller, fxt_runnable_factory, fxt_job
    ):
        """Test execution of jobs with different behaviors (success, failure, cancellation)."""
        jobs = []

        # Create different types of jobs
        success_job = fxt_job()
        failure_job = fxt_job()
        slow_job = fxt_job()

        jobs.extend([success_job, failure_job, slow_job])

        for job in jobs:
            await fxt_job_queue.submit(job)

        call_count = 0

        def mock_factory(_: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockRunnable(behavior=RunnableBehaviour.SUCCESS)
            if call_count == 2:
                return MockRunnable(behavior=RunnableBehaviour.FAILURE)
            return MockRunnable(behavior=RunnableBehaviour.SLOW, execution_time=0.02)

        fxt_runnable_factory.side_effect = mock_factory

        await fxt_job_controller.start()

        try:
            # Wait for success and failure jobs to complete
            await self._wait_for_job_status(success_job, JobStatus.DONE, timeout=3.0)
            await self._wait_for_job_status(failure_job, JobStatus.FAILED, timeout=3.0)

            # Wait for slow job to start then cancel it
            await self._wait_for_job_status(slow_job, JobStatus.RUNNING, timeout=2.0)
            fxt_job_queue.cancel(slow_job.id)
            await self._wait_for_job_status(slow_job, JobStatus.CANCELLED, timeout=2.0)

            # Verify final states
            assert success_job.status == JobStatus.DONE
            assert failure_job.status == JobStatus.FAILED
            assert slow_job.status == JobStatus.CANCELLED

        finally:
            await fxt_job_controller.stop()

    @pytest.mark.asyncio
    async def test_supervisor_loop_error_recovery(self, fxt_job_queue, fxt_runner_factory, fxt_job):
        """Test that supervisor loop handles various errors gracefully and continues operating."""
        job_controller = JobController(fxt_job_queue, fxt_runner_factory, max_parallel_jobs=1)

        # Track supervisor loop iterations and errors
        error_count = 0

        # Create jobs for testing recovery
        recovery_job = fxt_job()
        post_error_job = fxt_job()

        # Mock runner factory that can simulate runner creation errors
        original_for_job = fxt_runner_factory.for_job

        def error_prone_runner_factory(job):
            nonlocal error_count
            if job == recovery_job and error_count == 0:
                error_count += 1
                raise RuntimeError("Simulated runner creation error")
            return original_for_job(job)

        fxt_runner_factory.for_job = error_prone_runner_factory

        await job_controller.start()

        try:
            # Submit job that will cause error during runner creation
            await fxt_job_queue.submit(recovery_job)

            # Wait briefly for error to be processed
            await asyncio.sleep(0.1)

            # Submit job that should succeed after error recovery
            await fxt_job_queue.submit(post_error_job)

            # Wait for the successful job to complete
            await self._wait_for_job_status(post_error_job, JobStatus.DONE, timeout=3.0)

            # Verify error was handled and supervisor continued
            assert error_count == 1, "Expected exactly one error during runner creation"
            assert post_error_job.status == JobStatus.DONE, "Job after error should complete successfully"
            assert recovery_job.status == JobStatus.PENDING, "Failed job should remain pending for retry"

        finally:
            await job_controller.stop()

    @pytest.mark.asyncio
    async def test_supervisor_loop_continuous_operation_after_errors(
        self, fxt_job_queue, fxt_runner_factory, fxt_runnable_factory, fxt_job
    ):
        """Test that supervisor continues processing jobs normally after handling errors."""
        job_controller = JobController(fxt_job_queue, fxt_runner_factory, max_parallel_jobs=1)

        # Create multiple jobs
        jobs_before_error = [fxt_job() for _ in range(2)]
        error_job = fxt_job()
        jobs_after_error = [fxt_job() for _ in range(2)]

        all_jobs = [*jobs_before_error, error_job, *jobs_after_error]

        # Submit all jobs
        for job in all_jobs:
            await fxt_job_queue.submit(job)

        # Mock factory that causes error for specific job
        call_count = 0

        def selective_factory(_: str):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Third job (error_job) will fail
                return MockRunnable(behavior=RunnableBehaviour.FAILURE)
            return MockRunnable(behavior=RunnableBehaviour.SUCCESS)

        fxt_runnable_factory.side_effect = selective_factory

        await job_controller.start()

        try:
            # Wait for all jobs to be processed
            for job in jobs_before_error + jobs_after_error:
                await self._wait_for_job_status(job, JobStatus.DONE, timeout=5.0)

            await self._wait_for_job_status(error_job, JobStatus.FAILED, timeout=5.0)

            # Verify continuous operation
            assert all(job.status == JobStatus.DONE for job in jobs_before_error), "Jobs before error should succeed"
            assert error_job.status == JobStatus.FAILED, "Error job should fail"
            assert all(job.status == JobStatus.DONE for job in jobs_after_error), "Jobs after error should succeed"

        finally:
            await job_controller.stop()

    @staticmethod
    async def _wait_for_job_status(job: Job, expected_status: JobStatus, timeout: float) -> None:
        """Helper method to wait for job to reach expected status."""
        async with asyncio.timeout(timeout):
            while job.status != expected_status:
                await asyncio.sleep(0.01)
