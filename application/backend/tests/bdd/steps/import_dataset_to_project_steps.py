# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import secrets
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
import requests
from behave import given, then, when
from behave.model import Table
from behave.runner import Context
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from app.api.schemas import ProjectView
from app.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)
from app.models import DatasetItemAnnotationStatus, Task, TaskType
from tests.bdd.utils import generate_random_image, import_dataset_to_project


@given("a dataset with labels {labels} exists")  # pyrefly: ignore
def step_dataset_with_labels_exists(context: Context, labels: str) -> None:
    """Create a dataset with the specified labels."""
    labels_list = ast.literal_eval(labels)
    label_categories: Categories = LabelCategories(labels=labels_list)
    context.labels = label_categories
    sample_type: Any = None
    project = cast(ProjectView, context.project)
    match project.task.task_type:
        case TaskType.DETECTION:
            sample_type = DetectionSample
        case TaskType.CLASSIFICATION:
            sample_type = ClassificationSample if project.task.exclusive_labels else MultilabelClassificationSample
        case TaskType.INSTANCE_SEGMENTATION:
            sample_type = InstanceSegmentationSample
    context.dataset = Dataset(sample_type, categories={"label": label_categories})


@given("the dataset contains the following image distribution:")  # pyrefly: ignore
def step_dataset_with_samples_exists(context: Context) -> None:
    """Add multiple random annotated images to the dataset specific subset."""
    dataset = cast(Dataset, context.dataset)
    task = cast(Task, context.task)
    labels = cast(LabelCategories, context.labels)
    images_dir = cast(Path, context.tmp_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for row in cast(Table, context.table):
        label_name = row["Label"]
        label_idx, _ = labels.find(label_name)

        if label_idx is None:
            raise ValueError(f"Label '{label_name}' definition not found in dataset categories")

        for subset in ["Training", "Validation", "Testing"]:
            count = int(row.get(subset, "0").strip())

            for i in range(count):
                buffer, filename = generate_random_image()
                image_path = images_dir / f"{subset}_{filename}"
                image_path.write_bytes(buffer.read())
                sample = None
                match task.task_type:
                    case TaskType.DETECTION:
                        x1, y1 = 10 + secrets.randbelow(50), 20 + secrets.randbelow(50)
                        x2, y2 = x1 + 80 + secrets.randbelow(100), y1 + 100 + secrets.randbelow(150)
                        sample = DetectionSample(
                            id=None,
                            image=LazyImage(image_path),
                            image_info=ImageInfo(width=640, height=480),
                            subset=Subset[subset.upper()],
                            user_reviewed=True,
                            label=np.array([label_idx]),
                            bboxes=np.array([[x1, y1, x2, y2]]),
                            confidence=None,
                        )
                    case TaskType.CLASSIFICATION:
                        if task.exclusive_labels:
                            sample = ClassificationSample(
                                id=None,
                                image=LazyImage(image_path),
                                image_info=ImageInfo(width=640, height=480),
                                subset=Subset[subset.upper()],
                                user_reviewed=True,
                                label=label_idx,
                                confidence=None,
                            )
                        else:
                            sample = MultilabelClassificationSample(
                                id=None,
                                image=LazyImage(image_path),
                                image_info=ImageInfo(width=640, height=480),
                                subset=Subset[subset.upper()],
                                user_reviewed=True,
                                label=np.array([label_idx]),
                                confidence=None,
                            )
                    case TaskType.INSTANCE_SEGMENTATION:
                        sample = InstanceSegmentationSample(
                            id=None,
                            image=LazyImage(image_path),
                            image_info=ImageInfo(width=640, height=480),
                            subset=Subset[subset.upper()],
                            user_reviewed=True,
                            label=np.array([label_idx]),
                            polygons=np.array(
                                [
                                    [
                                        [10 + secrets.randbelow(50), 20 + secrets.randbelow(50)],
                                        [60 + secrets.randbelow(50), 20 + secrets.randbelow(50)],
                                        [60 + secrets.randbelow(50), 120 + secrets.randbelow(150)],
                                        [10 + secrets.randbelow(50), 120 + secrets.randbelow(150)],
                                    ]
                                ]
                            ),
                            confidence=None,
                        )
                dataset.append(sample)


@given("the dataset is ready for import in staging directory")  # pyrefly: ignore
def step_dataset_is_ready_for_import(context: Context) -> None:
    """Export the dataset to a staging directory to prepare for import."""
    dataset = cast(Dataset, context.dataset)
    staged_dataset_id = uuid4()
    staging_dir = cast(Path, context.tmp_path) / "data" / "staged_datasets" / str(staged_dataset_id)
    export_dataset(dataset, staging_dir / "dataset")
    context.staged_dataset_id = staged_dataset_id


@when("I import the dataset with label mappings:")  # pyrefly: ignore
def step_import_dataset_with_mapping(context: Context) -> None:
    """Import the dataset with label mappings specified in the table."""
    labels_mapping: dict[str, str | None] = {}
    for row in cast(Table, context.table):
        source = row["Source Label"]
        target = row["Target Label"]
        if target.lower() in ("none", ""):
            labels_mapping[source] = None
        else:
            labels_mapping[source] = target
    project = cast(ProjectView, context.project)
    staged_dataset_id = cast(UUID, context.staged_dataset_id)
    import_dataset_to_project(str(context.base_url), str(project.id), str(staged_dataset_id), labels_mapping)


@when("I import the dataset")  # pyrefly: ignore
def step_import_dataset(context: Context) -> None:
    """Import the dataset with label mappings specified in the table."""
    project = cast(ProjectView, context.project)
    staged_dataset_id = cast(UUID, context.staged_dataset_id)
    import_dataset_to_project(str(context.base_url), str(project.id), str(staged_dataset_id))


@then('the project contains {count:d} annotated images labeled "{label_name}"')  # pyrefly: ignore
def step_project_contains_annotated_images(context: Context, count: int, label_name: str) -> None:
    """Verify that the project dataset contains the expected number of annotated images with the specified label."""
    project = cast(ProjectView, context.project)
    label_ids = ",".join(str(label.id) for label in project.task.labels if label.name == label_name)
    # Verify items existing with the label
    response = requests.get(f"{context.base_url}/api/projects/{project.id}/dataset/items?labels={label_ids}")
    actual_item_count = response.json()["pagination"]["total"]
    assert actual_item_count == count, f"Expected {count} items with label '{label_name}', got {actual_item_count}"
    # Verify media existing with the label
    response = requests.get(f"{context.base_url}/api/projects/{project.id}/dataset/media?labels={label_ids}")
    actual_media_count = response.json()["pagination"]["total"]
    assert actual_media_count == count, f"Expected {count} media with label '{label_name}', got {actual_media_count}"


@then("the project contains {count:d} unannotated images")  # pyrefly: ignore
def step_project_contains_unannotated_images(context: Context, count: int) -> None:
    """Verify that the project dataset contains the expected number of annotated images with the specified label."""
    project = cast(ProjectView, context.project)
    # Verify items existing without annotations
    response = requests.get(
        f"{context.base_url}/api/projects/{project.id}/dataset/items?"
        f"annotation_status={DatasetItemAnnotationStatus.UNANNOTATED}"
    )
    actual_item_count = response.json()["pagination"]["total"]
    assert actual_item_count == count, f"Expected {count} unannotated items, got {actual_item_count}"
    # Verify media existing without annotations
    response = requests.get(
        f"{context.base_url}/api/projects/{project.id}/dataset/media?"
        f"annotation_status={DatasetItemAnnotationStatus.UNANNOTATED}"
    )
    actual_media_count = response.json()["pagination"]["total"]
    assert actual_media_count == count, f"Expected {count} unannotated media, got {actual_media_count}"
