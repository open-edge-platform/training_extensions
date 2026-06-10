# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import io
import secrets
from pathlib import Path
from typing import cast

from behave import given, then, when
from behave.model import Table
from behave.runner import Context
from datumaro.experimental import LazyImage, LazyVideoFrame
from datumaro.experimental.export_import import import_dataset
from requests import Session

from app.api.schemas import ProjectView
from app.api.schemas.jobs.dataset_export import ExportDatasetMetadata
from app.models import DatasetFormat, TaskType
from tests.bdd.utils import export_dataset, generate_random_image, generate_random_video


@given('A {task_type} project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
@given('An {task_type} project "{project_name}" with labels {labels} exists')  # pyrefly: ignore
def step_project_exists(context: Context, task_type: TaskType, project_name: str, labels: str) -> None:
    session = cast(Session, context.session)
    labels_list = ast.literal_eval(labels)
    project_body = {
        "name": project_name,
        "task": {
            "task_type": TaskType.CLASSIFICATION if task_type == "multilabel" else task_type,
            "exclusive_labels": task_type == TaskType.CLASSIFICATION,
            "labels": [{"name": label, "color": f"#{secrets.token_hex(3)}"} for label in labels_list],
        },
    }
    response = session.post(f"{context.base_url}/api/projects", json=project_body)
    project = ProjectView.model_validate(response.json())
    context.project = project
    context.task = project.task


@given("the project contains the following image distribution:")  # pyrefly: ignore
def step_project_dataset_has_images(context: Context) -> None:
    """
    Add multiple random images with specific labels to specific subsets based on a table.
    Table format expected:
      | Label      | Training | Validation |
      | Dog        | 10       | 5          |
    """
    project = cast(ProjectView, context.project)
    tmp_path = cast(Path, context.tmp_path)
    session = cast(Session, context.session)
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
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
                image_path = generate_random_image(output_path=images_dir, suffix="upload")
                buffer = io.BytesIO()
                with open(image_path, "rb") as image_file:
                    buffer.write(image_file.read())
                buffer.seek(0)
                files = {"file": (image_path.name, buffer, "image/jpeg")}
                media_response = session.post(
                    f"{context.base_url}/api/projects/{project.id}/dataset/media", files=files
                )
                media_id = media_response.json()["id"]

                # 2. Add annotation based on task type
                annotation_body = {
                    "subset": subset.lower(),
                }
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
                response = session.post(
                    f"{context.base_url}/api/projects/{project.id}/dataset/media/{media_id}/annotations",
                    json=annotation_body,
                )
                assert response.status_code == 201, (
                    f"Failed to add annotation, status code: {response.status_code}, response: {response.text}"
                )


@given("the project contains the following video frame distribution:")  # pyrefly: ignore
def step_project_dataset_has_video_frames(context: Context) -> None:
    """
    Add multiple random video frames with specific labels to specific subsets based on a Gherkin table.
    Table format expected:
      | Label      | Training | Validation |
      | Dog        | 10       | 5          |
    """
    tmp_path = cast(Path, context.tmp_path)
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    # 1. Upload random video
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = generate_random_video(video_dir)
    buffer = io.BytesIO()
    with open(video_path, "rb") as video_file:
        buffer.write(video_file.read())
    buffer.seek(0)
    files = {"file": (video_path.name, buffer, "image/jpeg")}
    media_response = session.post(f"{context.base_url}/api/projects/{project.id}/dataset/media", files=files)
    video_id = media_response.json()["id"]
    label_name_to_uuid = {label.name: str(label.id) for label in project.task.labels}
    frame_idx = 0
    for row in cast(Table, context.table):
        label_name = row["Label"]

        if label_name not in label_name_to_uuid:
            raise ValueError(f"Label '{label_name}' not found in project labels: {list(label_name_to_uuid.keys())}")

        label_id = label_name_to_uuid[label_name]

        for subset in ["Training", "Validation", "Testing"]:
            count = int(row.get(subset, "0").strip())
            for i in range(count):
                # 1. Add annotation based on task type
                annotation_body = {
                    "subset": subset.lower(),
                }
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
                response = session.post(
                    f"{context.base_url}/api/projects/{project.id}/dataset/media/{video_id}/annotations?"
                    f"frame_index={frame_idx}",
                    json=annotation_body,
                )
                assert response.status_code == 201, (
                    f"Failed to add annotation, status code: {response.status_code}, response: {response.text}"
                )
                frame_idx += 1


@when("I export the project dataset in {export_format} format with filters={filters}")  # pyrefly: ignore
def step_export_dataset(context: Context, export_format: str, filters: str) -> None:
    project = cast(ProjectView, context.project)
    export_format = DatasetFormat(export_format.lower())
    job = export_dataset(cast(Session, context.session), str(context.base_url), str(project.id), export_format, filters)
    context.export_format = export_format
    context.dataset_id = cast(ExportDatasetMetadata, job.metadata).dataset_id


@then("the staged dataset archive {archive_name} should exist")  # pyrefly: ignore
def step_staged_dataset_archive_exists(context: Context, archive_name: str) -> None:
    session = cast(Session, context.session)
    response = session.get(f"{context.base_url}/api/staged_datasets/{context.dataset_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )


@then("the staged dataset with name={dataset_name} has {count:d} {media_type}")  # pyrefly: ignore
def step_staged_dataset_has_items(context: Context, dataset_name: str, count: int, media_type: str) -> None:
    export_format = cast(DatasetFormat, context.export_format)
    dataset_path = cast(Path, context.tmp_path) / "data" / "staged_datasets" / str(context.dataset_id) / dataset_name
    actual_count = 0
    if export_format != DatasetFormat.GETI:
        if media_type == "video frames":
            return
        if media_type == "images":
            dataset = import_dataset(dataset_path)
            actual_count = len(dataset)
    else:
        dataset = import_dataset(dataset_path)
        media_class = LazyVideoFrame if media_type == "video frames" else LazyImage
        actual_count = sum(1 for sample in dataset if isinstance(sample.media, media_class))

    assert actual_count == count, f"Expected {count} {media_type} in dataset, but found {actual_count}"
