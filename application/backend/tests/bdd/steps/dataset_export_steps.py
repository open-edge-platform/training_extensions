# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import secrets
from pathlib import Path
from typing import cast

import requests
from behave import given, then, when
from behave.model import Table
from behave.runner import Context

from app.api.schemas import ProjectView
from app.api.schemas.jobs.dataset_export import ExportDatasetMetadata
from app.models import DatasetFormat, TaskType
from tests.bdd.utils import export_dataset, generate_random_image, import_dataset_by_format


@given('A {task_type} project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
@given('An {task_type} project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
def step_project_exists(context: Context, task_type: TaskType, project_name: str, labels: str) -> None:
    labels_list = ast.literal_eval(labels)
    project_body = {
        "name": project_name,
        "task": {
            "task_type": TaskType.CLASSIFICATION if task_type == "multilabel" else task_type,
            "exclusive_labels": task_type == TaskType.CLASSIFICATION,
            "labels": [{"name": label, "color": f"#{secrets.token_hex(3)}"} for label in labels_list],
        },
    }
    response = requests.post(f"{context.base_url}/api/projects", json=project_body)
    context.project = ProjectView.model_validate(response.json())


@given("the project contains the following image distribution:")  # pyrefly: ignore
def step_project_dataset_has_images(context: Context) -> None:
    """
    Add multiple random images with specific labels to specific subsets based on a Gherkin table.
    Table format expected:
      | Label      | Training | Validation |
      | Dog        | 10       | 5          |
    """
    project = cast(ProjectView, context.project)
    label_name_to_uuid = {label.name: str(label.id) for label in project.task.labels}

    for row in cast(Table, context.table):
        label_name = row["Label"]

        if label_name not in label_name_to_uuid:
            raise ValueError(f"Label '{label_name}' not found in project labels: {list(label_name_to_uuid.keys())}")

        label_id = label_name_to_uuid[label_name]

        for subset in ["Training", "Validation", "Testing"]:
            count = int(row.get(subset, "0").strip())
            for i in range(count):
                # 1. Upload random image
                buffer, filename = generate_random_image()
                files = {"file": (filename, buffer, "image/jpeg")}
                media_response = requests.post(
                    f"{context.base_url}/api/projects/{project.id}/dataset/media", files=files
                )
                media_id = media_response.json()["id"]

                # 2. Add annotation based on task type
                annotation_body = {}
                match project.task.task_type:
                    case TaskType.CLASSIFICATION:
                        annotation_body = {
                            "annotations": [
                                {
                                    "labels": [{"id": label_id}],
                                    "shape": {"type": "full_image"},
                                },
                            ],
                        }
                    case TaskType.DETECTION:
                        annotation_body = {
                            "annotations": [
                                {
                                    "labels": [{"id": label_id}],
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
                                    "labels": [{"id": label_id}],
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
                response = requests.post(
                    f"{context.base_url}/api/projects/{project.id}/dataset/media/{media_id}/annotations",
                    json=annotation_body,
                )
                assert response.status_code == 201, (
                    f"Failed to add annotation, status code: {response.status_code}, response: {response.text}"
                )

                # 3. Assign subset
                response = requests.patch(
                    f"{context.base_url}/api/projects/{project.id}/dataset/items/{media_id}/subset",
                    json={"subset": subset.lower()},
                )
                assert response.status_code == 200, (
                    f"Failed to assign subset, status code: {response.status_code}, response: {response.text}"
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
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )


@then("the staged dataset with name={dataset_name} has {count:d} images")  # pyrefly: ignore
def step_staged_dataset_has_items(context: Context, dataset_name: str, count: int) -> None:
    dataset_format = cast(DatasetFormat, context.export_format)
    dataset_path = cast(Path, context.tmp_path) / "data" / "staged_datasets" / str(context.dataset_id) / dataset_name
    dataset = import_dataset_by_format(dataset_path, dataset_format)
    actual_count = len(dataset)
    assert actual_count == count, f"Expected {count} images in dataset, but found {actual_count}"
