# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from starlette import status

from app.api.dependencies import get_data_dir, get_job_dir, get_job_queue
from app.core.jobs import Job, JobParams, JobQueue, JobStatus
from app.core.jobs.control_plane import CancellationResult
from app.main import app
from app.models import TaskType
from app.schemas import ProjectView
from app.schemas.job import JobRequest, JobType, TrainingRequestParams
from app.schemas.project import TaskView


@pytest.fixture
def fxt_jobs_queue() -> Mock:
    jobs_queue = Mock(spec=JobQueue)
    app.dependency_overrides[get_job_queue] = lambda: jobs_queue
    return jobs_queue


@pytest.fixture
def fxt_job() -> Callable[[UUID | None, JobStatus, float], Job]:
    def job_factory(
        job_id: UUID | None = None, job_status: JobStatus = JobStatus.RUNNING, progress: float = 50.0
    ) -> Job:
        return Job(
            id=job_id or uuid4(),
            status=job_status,
            progress=100.0 if job_status >= JobStatus.DONE else progress,
            job_type=JobType.TRAIN,
            params=JobParams(),
        )

    return job_factory


class TestJobEndpoints:
    def test_submit_train_job(self, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service):
        app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=ProjectView)
        project.task = Mock(spec=TaskView)
        project.task.task_type = TaskType.CLASSIFICATION
        fxt_project_service.get_project_by_id.return_value = project
        job_request = JobRequest(
            project_id=uuid4(),
            job_type=JobType.TRAIN,
            parameters=TrainingRequestParams(
                model_architecture_id="YOLOv8",
                parent_model_revision_id=uuid4(),
            ),
        )

        response = fxt_client.post("/api/jobs", json=job_request.model_dump(mode="json"))

        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json()["job_id"]
        fxt_project_service.get_project_by_id.assert_called_once_with(job_request.project_id)
        fxt_jobs_queue.submit.assert_called_once()
        assert fxt_jobs_queue.submit.call_args[0][0].params.model_architecture_id == "YOLOv8"
        assert fxt_jobs_queue.submit.call_args[0][0].params.task_type == TaskType.CLASSIFICATION

    def test_list_jobs(self, fxt_client, fxt_jobs_queue, fxt_job):
        fxt_jobs_queue.list_all.return_value = [fxt_job(), fxt_job()]

        response = fxt_client.get("/api/jobs")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_jobs_queue.list_all.assert_called_once()

    def test_get_job(self, fxt_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = fxt_job(job_id)

        response = fxt_client.get(f"/api/jobs/{job_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["job_id"] == str(job_id)
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    def test_get_job_not_found(self, fxt_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = None

        response = fxt_client.get(f"/api/jobs/{job_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.parametrize(
        "job_id, cancellation_result, expected_status",
        [
            (uuid4(), CancellationResult.PENDING_CANCELLED, status.HTTP_202_ACCEPTED),
            (uuid4(), CancellationResult.RUNNING_CANCELLING, status.HTTP_202_ACCEPTED),
            (uuid4(), CancellationResult.IGNORE_CANCEL, status.HTTP_409_CONFLICT),
            (uuid4(), CancellationResult.NOT_FOUND, status.HTTP_404_NOT_FOUND),
        ],
    )
    def test_cancel_job(self, job_id, cancellation_result, expected_status, fxt_client, fxt_jobs_queue, fxt_job):
        fxt_jobs_queue.cancel.return_value = fxt_job(job_id), cancellation_result

        response = fxt_client.post(f"/api/jobs/{job_id}:cancel")

        assert response.status_code == expected_status
        if expected_status == status.HTTP_202_ACCEPTED:
            assert response.json()["job_id"] == str(job_id)
        fxt_jobs_queue.cancel.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_status_not_found(self, fxt_async_client, fxt_jobs_queue):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = None

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/status")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_status_stops_when_done(self, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        # Simulate a job that completes after a few updates
        job_running = fxt_job(job_id, JobStatus.RUNNING, progress=50.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)

        # First call returns running job, subsequent calls return done job
        fxt_jobs_queue.get.side_effect = [job_running, job_running, job_done, None]

        async def stream_test():
            events = []
            async with fxt_async_client.stream("GET", f"/api/jobs/{job_id}/status") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line)
            return events

        events = await asyncio.wait_for(stream_test(), 3)
        assert len(events) == 2
        # Verify the stream contains job status updates
        assert '"status":"RUNNING"' in events[0]
        assert '"status":"DONE"' in events[1]

    @pytest.mark.asyncio
    async def test_stream_job_status_yields_only_changed_updates(self, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_v1 = fxt_job(job_id, JobStatus.RUNNING, progress=25.0)
        job_v2 = fxt_job(job_id, JobStatus.RUNNING, progress=75.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)

        # Return same job twice (no change), then changed job, then done
        fxt_jobs_queue.get.side_effect = [job_v1, job_v1, job_v2, job_done, None]

        async def stream_test():
            events = []
            async with fxt_async_client.stream("GET", f"/api/jobs/{job_id}/status") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line)
            return events

        events = await asyncio.wait_for(stream_test(), 3)
        # Should get at least 2 events (initial and one change)
        assert len(events) == 3
        assert '"progress":25.0' in events[0]
        assert '"progress":75.0' in events[1]
        assert '"progress":100' in events[2]

    @pytest.mark.asyncio
    async def test_stream_job_logs_yields_log_lines(self, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job_v1 = fxt_job(job_id, JobStatus.RUNNING, progress=25.0)
        job_v2 = fxt_job(job_id, JobStatus.RUNNING, progress=75.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)
        # Create a log file with some content using the job's log_file property
        log_file = job_dir / job_v1.log_file
        log_file.write_text("Line 1\nLine 2\nLine 3\n")
        fxt_jobs_queue.get.side_effect = [job_v1, job_v1, job_v2, job_done, None]

        app.dependency_overrides[get_job_dir] = lambda: job_dir

        async def stream_test():
            events = []
            async with fxt_async_client.stream("GET", f"/api/jobs/{job_id}/logs") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line)
            return events

        events = await asyncio.wait_for(stream_test(), 2)
        assert len(events) == 3
        assert "Line 1" in events[0]
        assert "Line 2" in events[1]
        assert "Line 3" in events[2]

    @pytest.mark.asyncio
    async def test_stream_job_logs_not_found(self, tmp_path, fxt_async_client, fxt_jobs_queue):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        fxt_jobs_queue.get.return_value = None

        app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_logs_completed(self, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job = fxt_job(job_id, JobStatus.DONE)
        fxt_jobs_queue.get.return_value = job

        app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "completed" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_stream_job_logs_file_not_found(self, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job = fxt_job(job_id, JobStatus.RUNNING)
        # Don't create the log file - it should not exist
        fxt_jobs_queue.get.return_value = job

        app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "log file not found" in response.json()["detail"].lower()
