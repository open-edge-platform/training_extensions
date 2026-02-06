# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import random
import secrets
import zipfile
from pathlib import Path
from typing import cast

import requests
from behave import given, then, when
from behave.runner import Context
from datumaro.experimental.data_formats.base import load_dataset
from datumaro.experimental.export_import import import_dataset

from app.api.schemas import ProjectView
from app.api.schemas.jobs import JobView
from app.core.jobs.models import JobStatus, JobType
from app.executors.dataset_export.exporter import get_dm_format
from app.models import DatasetFormat, TaskType
from tests.bdd.images import generate_random_image
from tests.bdd.parsers import parse_sse_events


@given('A "{task_type}" project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
@given('An "{task_type}" project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
def step_project_exists(context: Context, task_type: TaskType, project_name: str, labels: str) -> None:
    labels_list = ast.literal_eval(labels)
    project_body = {
        "name": project_name,
        "task": {
            "task_type": task_type,
            "exclusive_labels": True,
            "labels": [{"name": label, "color": f"#{secrets.token_hex(3)}"} for label in labels_list],
        },
    }
    response = requests.post(f"{context.base_url}/api/projects", json=project_body)
    context.project = ProjectView.model_validate(response.json())


@given('the project dataset has {count:d} unannotated images in subset "{subset}"')  # pyrefly: ignore
def step_dataset_has_unannotated_images(context: Context, count: int, subset: str) -> None:
    """Add multiple random unannotated images to the dataset."""
    project = cast(ProjectView, context.project)
    for _ in range(count):
        # Upload random image
        buffer, filename, _ = generate_random_image()
        files = {"file": (filename, buffer, "image/jpeg")}
        media_response = requests.post(f"{context.base_url}/api/projects/{project.id}/dataset/media", files=files)
        dataset_item_id = media_response.json()["id"]

        # Assign subset
        requests.patch(
            f"{context.base_url}/api/projects/{project.id}/dataset/items/{dataset_item_id}/subset",
            json={"subset": subset},
        )


@given('the project dataset has {count:d} images with annotations in subset "{subset}"')  # pyrefly: ignore
def step_dataset_has_annotated_images(context: Context, count: int, subset: str) -> None:
    """Add multiple random unannotated images to the dataset specific subset."""
    project = cast(ProjectView, context.project)
    for i in range(count):
        # Upload random image
        buffer, filename, _ = generate_random_image()
        files = {"file": (filename, buffer, "image/jpeg")}
        media_response = requests.post(f"{context.base_url}/api/projects/{project.id}/dataset/media", files=files)
        dataset_item_id = media_response.json()["id"]

        # Set annotations
        label_ids = [str(label.id) for label in project.task.labels]
        annotation_body = {}
        match project.task.task_type:
            case TaskType.CLASSIFICATION:
                if len(label_ids) < 2:
                    raise ValueError("At least 2 labels are required for this step")
                annotation_body = {
                    "annotations": [
                        {
                            "labels": [{"id": label_ids[0]}] if i % 2 == 0 else [{"id": label_ids[1]}],
                            "shape": {"type": "full_image"},
                        },
                    ],
                }
            case TaskType.DETECTION:
                annotation_body = {
                    "annotations": [
                        {
                            "labels": [{"id": random.choice(label_ids)}],
                            "shape": {
                                "type": "rectangle",
                                "x": 10 + secrets.randbelow(50),
                                "y": 20 + secrets.randbelow(50),
                                "width": 80 + secrets.randbelow(100),
                                "height": 100 + secrets.randbelow(150),
                            },
                        }
                    ]
                }
        requests.post(
            f"{context.base_url}/api/projects/{project.id}/dataset/items/{dataset_item_id}/annotations",
            json=annotation_body,
        )

        # Assign subset
        requests.patch(
            f"{context.base_url}/api/projects/{project.id}/dataset/items/{dataset_item_id}/subset",
            json={"subset": subset},
        )


@when("I export the project dataset in {export_format} format with filters={filters}")  # pyrefly: ignore
def step_export_dataset(context: Context, export_format: str, filters: str) -> None:
    project = cast(ProjectView, context.project)
    job_body = {
        "job_type": JobType.EXPORT_DATASET,
        "project_id": str(project.id),
        "parameters": {
            "export_format": DatasetFormat(export_format.lower()),
            "filters": json.loads(filters),
        },
    }
    response = requests.post(f"{context.base_url}/api/jobs", json=job_body)
    job = JobView.model_validate(response.json())
    context.export_format = DatasetFormat(export_format.lower())

    with requests.get(
        f"{context.base_url}/api/jobs/{job.job_id}/status", stream=True, headers={"Accept": "text/event-stream"}
    ) as stream_response:
        for job_data in parse_sse_events(stream_response):
            job = JobView.model_validate(job_data)
            if job.status in (JobStatus.DONE.name, JobStatus.FAILED.name):
                context.job = job
                break

    job = cast(JobView, context.job)
    assert job.status == JobStatus.DONE.name, f"Expected job to be DONE, but got {job.status}, error: {job.error}"


@then("the staged dataset archive {archive_name} should exist")  # pyrefly: ignore
def step_staged_dataset_archive_exists(context: Context, archive_name: str) -> None:
    staged_datasets_dir = cast(Path, context.tmp_path) / "data" / "staged_datasets"

    matching_archives = [f for f in staged_datasets_dir.glob("**/*.zip") if f.name.endswith(archive_name)]

    assert len(matching_archives) > 0, (
        f"No archive ending with '{archive_name}' found in {staged_datasets_dir}. "
        f"Available archives: {list(staged_datasets_dir.glob('*.zip'))}"
    )

    context.staged_dataset_path = staged_datasets_dir / matching_archives[0]


@then("the exported dataset has {count:d} images")  # pyrefly: ignore
def step_exported_dataset_has_items(context: Context, count: int) -> None:
    dataset = None
    export_format = context.export_format
    dataset_path = cast(Path, context.staged_dataset_path)
    match export_format:
        case DatasetFormat.GETI:
            dataset = import_dataset(dataset_path)
        case DatasetFormat.YOLO:
            extract_dir = dataset_path.with_suffix("")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            dataset = load_dataset(
                data_format=get_dm_format(export_format),
                root_dir=str(extract_dir),
            )
        case _:
            raise Exception(f"Unknown export format: {export_format}")

    actual_count = len(dataset)
    assert actual_count == count, f"Expected {count} images in dataset, but found {actual_count}"
