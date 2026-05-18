# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import time
from uuid import UUID

import requests

from app.api.schemas.jobs import JobView
from app.core.jobs.models import JobStatus, JobType
from app.models import DatasetFormat, TaskType
from tests.bdd.utils.parsers import parse_sse_events

# Overall budget for waiting on a single job to reach a terminal state.
# The job status endpoint streams updates over SSE; if the stream is closed
# prematurely (e.g. due to a transient network/proxy issue) we reconnect
# until the job is DONE/FAILED or this budget is exhausted.
_JOB_WAIT_TIMEOUT_SECONDS = 600
_JOB_RECONNECT_SLEEP_SECONDS = 1


def expect_job_accepted(response: requests.Response) -> JobView:
    assert response.status_code == 202, (
        f"Expected job to be ACCEPTED, but got {response.status_code}, response: {response.text}"
    )
    return JobView.model_validate(response.json())


def wait_for_job_completion(base_url: str, job_id: UUID) -> JobView:
    """Wait for a job to reach a terminal state by consuming its SSE status stream.

    If the SSE stream ends before the job reaches a terminal state (e.g. due to
    a transient network/proxy disconnect), the function reconnects and keeps
    waiting, falling back to a regular GET on the job resource so that the test
    does not flake on benign disconnections. A total timeout bounds the wait.
    """
    terminal_statuses = (JobStatus.DONE.name, JobStatus.FAILED.name)
    job: JobView | None = None
    deadline = time.monotonic() + _JOB_WAIT_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        with requests.get(
            f"{base_url}/api/jobs/{job_id}/status",
            stream=True,
            headers={"Accept": "text/event-stream"},
        ) as stream_response:
            for job_data in parse_sse_events(stream_response):
                job = JobView.model_validate(job_data)
                if job.status in terminal_statuses:
                    break

        if job is not None and job.status in terminal_statuses:
            break

        # SSE stream ended before reaching a terminal state. Re-check the job
        # status directly in case the terminal event was missed, then reconnect.
        poll_response = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=10)
        if poll_response.status_code == 200:
            job = JobView.model_validate(poll_response.json())
            if job.status in terminal_statuses:
                break

        time.sleep(_JOB_RECONNECT_SLEEP_SECONDS)

    assert job is not None, f"Did not receive any status update for job {job_id}"
    assert job.status == JobStatus.DONE.name, f"Expected job to be DONE, but got {job.status}, error: {job.error}"
    return job


def export_dataset(base_url: str, project_id: str, export_format: DatasetFormat, filters: str | None = None) -> JobView:
    job_body = {
        "job_type": JobType.EXPORT_DATASET,
        "project_id": project_id,
        "parameters": {
            "export_format": export_format,
        },
    }
    if filters:
        job_body["parameters"]["filters"] = json.loads(filters)
    response = requests.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(base_url, job.job_id)


def prepare_dataset(base_url: str, staged_dataset_id: str) -> JobView:
    job_body = {
        "job_type": JobType.PREPARE_DATASET_FOR_IMPORT,
        "staged_dataset_id": staged_dataset_id,
    }
    response = requests.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(base_url, job.job_id)


def import_dataset_to_project(
    base_url: str, project_id: str, staged_dataset_id: str, labels_mapping: dict[str, str | None] | None = None
) -> JobView:
    job_body = {
        "job_type": JobType.IMPORT_DATASET_TO_PROJECT,
        "project_id": project_id,
        "staged_dataset_id": staged_dataset_id,
        "parameters": {},
    }
    if labels_mapping is not None:
        job_body["parameters"] = {"labels_mapping": labels_mapping}
    response = requests.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(base_url, job.job_id)


def import_dataset_as_new_project(
    base_url: str,
    project_name: str,
    staged_dataset_id: str,
    labels: list[str],
    task_type: TaskType,
    exclusive_labels: bool = False,
    include_unannotated: bool = True,
) -> JobView:
    job_body = {
        "job_type": JobType.IMPORT_DATASET_AS_NEW_PROJECT,
        "staged_dataset_id": staged_dataset_id,
        "parameters": {
            "project": {
                "name": project_name,
                "task_type": task_type,
                "exclusive_labels": exclusive_labels,
            },
            "filters": {
                "labels": labels,
                "include_unannotated": include_unannotated,
            },
        },
    }
    response = requests.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(base_url, job.job_id)
