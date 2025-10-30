# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from starlette import status

from app.api.dependencies import get_job_queue
from app.core.jobs import Job, JobParams, JobQueue, JobStatus
from app.core.jobs.control_plane import CancellationResult
from app.core.models import TaskType
from app.main import app
from app.schemas import ProjectView
from app.schemas.job import JobRequest, JobType, JobView, TrainingRequestParams
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
    def test_submit_train_job(self, fxt_client, fxt_jobs_queue, fxt_project_service):
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

    def test_stream_job_status(self, fxt_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job = fxt_job(job_id, JobStatus.DONE)
        fxt_jobs_queue.get.return_value = job

        with fxt_client.stream("GET", f"/api/jobs/{job_id}/status") as response:
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream"

            for line in response.iter_lines():
                assert line == JobView.of(job).model_dump_json()
                break
