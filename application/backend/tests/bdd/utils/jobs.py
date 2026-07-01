# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from uuid import UUID

import requests

from app.api.schemas.jobs import JobView
from app.core.jobs.models import JobStatus, JobType
from app.models import DatasetFormat, TaskType
from tests.bdd.utils.parsers import parse_sse_events


def expect_job_accepted(response: requests.Response) -> JobView:
    assert response.status_code == 202, (
        f"Expected job to be ACCEPTED, but got {response.status_code}, response: {response.text}"
    )
    return JobView.model_validate(response.json())


def wait_for_job_completion(base_url: str, job_id: UUID) -> JobView:
    with requests.get(
        f"{base_url}/api/jobs/{job_id}/status", stream=True, headers={"Accept": "text/event-stream"}
    ) as stream_response:
        for job_data in parse_sse_events(stream_response):
            job = JobView.model_validate(job_data)
            if job.status in (JobStatus.DONE.name, JobStatus.FAILED.name):
                break

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
