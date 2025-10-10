# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from starlette import status

from app.schemas.job import JobRequest, JobType, TrainingRequest


class TestJobEndpoints:
    def test_submit_train_job(self, fxt_client):
        job_request = JobRequest(
            project_id=uuid4(),
            job_type=JobType.TRAIN,
            parameters=TrainingRequest(
                model_architecture_id="YOLOv8",
                parent_model_revision_id=uuid4(),
            ),
        )
        response = fxt_client.post("/api/jobs", json=job_request.model_dump(mode="json"))
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json()["job_id"]

    def test_list_jobs(self, fxt_client):
        response = fxt_client.get("/api/jobs")
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 1

    def test_get_job(self, fxt_client):
        job_id = uuid4()
        response = fxt_client.get(f"/api/jobs/{job_id}")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"job_id": str(job_id)}

    def test_cancel_job(self, fxt_client):
        job_id = uuid4()
        response = fxt_client.post(f"/api/jobs/{job_id}:cancel")
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json() == {"job_id": str(job_id)}

    @pytest.mark.skip(reason="SSE endpoint needs proper implementation")
    def test_stream_job_status(self, fxt_client):
        job_id = uuid4()

        with fxt_client.stream("GET", f"/api/jobs/{job_id}/status") as response:
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream"

            for line in response.iter_lines():
                assert line == f"Hey there from {job_id}"
                break
