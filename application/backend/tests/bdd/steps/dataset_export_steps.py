# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import random
import secrets
from pathlib import Path
from typing import cast

import requests
from behave import given, then, when
from behave.runner import Context

from app.api.schemas import ProjectView
from app.api.schemas.jobs.dataset_export import ExportDatasetMetadata
from app.models import DatasetFormat, TaskType
from tests.bdd.utils import export_dataset, generate_random_image, import_dataset_by_format


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
            case TaskType.INSTANCE_SEGMENTATION:
                annotation_body = {
                    "annotations": [
                        {
                            "labels": [{"id": random.choice(label_ids)}],
                            "shape": {
                                "type": "polygon",
                                "points": [
                                    {"x": 10 + secrets.randbelow(50), "y": 20 + secrets.randbelow(50)},
                                    {"x": 60 + secrets.randbelow(50), "y": 20 + secrets.randbelow(50)},
                                    {"x": 60 + secrets.randbelow(50), "y": 120 + secrets.randbelow(150)},
                                    {"x": 10 + secrets.randbelow(50), "y": 120 + secrets.randbelow(150)},
                                ],
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
    export_format = DatasetFormat(export_format.lower())
    context.export_format = export_format
    job = export_dataset(str(context.base_url), str(project.id), export_format, filters)
    context.dataset_id = cast(ExportDatasetMetadata, job.metadata).dataset_id


@then("the staged dataset archive {archive_name} should exist")  # pyrefly: ignore
def step_staged_dataset_archive_exists(context: Context, archive_name: str) -> None:
    response = requests.get(f"{context.base_url}/api/staged_datasets/{context.dataset_id}")
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"


@then("the staged dataset with name={dataset_name} has {count:d} images")  # pyrefly: ignore
def step_staged_dataset_has_items(context: Context, dataset_name: str, count: int) -> None:
    dataset_format = cast(DatasetFormat, context.export_format)
    dataset_path = cast(Path, context.tmp_path) / "data" / "staged_datasets" / str(context.dataset_id) / dataset_name
    dataset = import_dataset_by_format(dataset_path, dataset_format)
    actual_count = len(dataset)
    assert actual_count == count, f"Expected {count} images in dataset, but found {actual_count}"
