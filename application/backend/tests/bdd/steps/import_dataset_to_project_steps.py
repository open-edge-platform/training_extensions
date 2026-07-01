# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
from functools import reduce
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from behave import given, then, when
from behave.model import Table
from behave.runner import Context
from datumaro.experimental import Dataset
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.export_import import export_dataset
from requests import Session

from app.api.schemas import ProjectView
from app.api.schemas.dataset import DatasetStatisticsView
from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.models import DatasetItemAnnotationStatus, Task, TaskType
from tests.bdd.utils import MediaProvider, SampleFactory, import_dataset_to_project

_SUBSETS = ("Training", "Validation", "Testing")


def get_path(data, path, default=None):
    """Utility function to safely access nested dictionary values using a dot-separated path."""
    try:
        return reduce(lambda d, key: d[key], path.split("."), data)
    except (KeyError, TypeError):
        return default


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
            sample_type = DetectionImportExportSample
        case TaskType.CLASSIFICATION:
            sample_type = (
                MulticlassClassificationImportExportSample
                if project.task.exclusive_labels
                else MultilabelClassificationImportExportSample
            )
        case TaskType.INSTANCE_SEGMENTATION:
            sample_type = InstanceSegmentationImportExportSample
    context.dataset = Dataset(sample_type, categories={"label": label_categories})


@given("the dataset contains the following {media_type} distribution:")  # pyrefly: ignore
def step_dataset_with_samples_exists(context: Context, media_type: str) -> None:
    """Add multiple random annotated images to the dataset specific subset."""
    dataset = cast(Dataset, context.dataset)
    labels = cast(LabelCategories, context.labels)
    media = MediaProvider(media_type, cast(Path, context.tmp_path))
    factory = SampleFactory(cast(Task, context.task))

    for row in cast(Table, context.table):
        label_name = row["Label"]
        label_idx, _ = labels.find(label_name)

        if label_idx is None:
            raise ValueError(f"Label '{label_name}' definition not found in dataset categories")

        for subset in _SUBSETS:
            for _ in range(int(row.get(subset, "0").strip())):
                lazy_media, media_info = media.next(subset)
                dataset.append(factory.build(label_idx, subset, lazy_media, media_info))


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
    import_dataset_to_project(
        cast(Session, context.session), str(context.base_url), str(project.id), str(staged_dataset_id), labels_mapping
    )


@when("I import the dataset")  # pyrefly: ignore
def step_import_dataset(context: Context) -> None:
    """Import the dataset with label mappings specified in the table."""
    project = cast(ProjectView, context.project)
    staged_dataset_id = cast(UUID, context.staged_dataset_id)
    import_dataset_to_project(
        cast(Session, context.session), str(context.base_url), str(project.id), str(staged_dataset_id)
    )


@then("the project statistics are:")  # pyrefly: ignore
def step_project_statistics_has(context: Context) -> None:
    """
    Verify that the project contains the expected number of annotated images and frames.

    Table format expected:
      | Metric                 | Count |
      | annotated_video_frames | 17    |
    """
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    response = session.get(f"{context.base_url}/api/projects/{project.id}/dataset/statistics")
    statistics = response.json()
    context.statistics = statistics

    json_paths = {
        "images": "media_counts.images",
        "annotated_images": "annotations_counts.annotated_images",
        "annotated_video_frames": "annotations_counts.annotated_video_frames",
    }

    for row in cast(Table, context.table):
        metric, count = row["Metric"], int(row["Count"])
        value = get_path(statistics, json_paths[metric])
        assert value == count, f"Expected {count} {metric}, got {value}"


@then("the project contains the following annotation instances:")  # pyrefly: ignore
def step_project_contains_annotated_media(context: Context) -> None:
    """
    Verify that the project contains the expected number of annotated images and frames.

    Table format expected:
      | Label           | Instances |
      | Chardonnay      | 11        |
    """
    project = cast(ProjectView, context.project)
    statistics = DatasetStatisticsView.model_validate(context.statistics)

    for row in cast(Table, context.table):
        label_name, expected_instances = row["Label"], int(row["Instances"])
        label_id = next(label.id for label in project.task.labels if label.name == label_name)
        instances_per_label = statistics.annotations_counts.instances_per_label
        actual_instances = next(
            (instance.instances for instance in instances_per_label if instance.label_id == label_id), 0
        )
        assert actual_instances == expected_instances, (
            f"Expected {expected_instances} instances with label {label_name}, got {actual_instances}"
        )


@then("the project contains {count:d} unannotated images")  # pyrefly: ignore
def step_project_contains_unannotated_images(context: Context, count: int) -> None:
    """Verify that the project dataset contains the expected number of annotated images with the specified label."""
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    # Verify items existing without annotations
    response = session.get(
        f"{context.base_url}/api/projects/{project.id}/dataset/items?"
        f"annotation_status={DatasetItemAnnotationStatus.MISSING_ANNOTATIONS}"
    )
    actual_item_count = response.json()["pagination"]["total"]
    assert actual_item_count == count, f"Expected {count} unannotated items, got {actual_item_count}"
    # Verify media existing without annotations
    response = session.get(
        f"{context.base_url}/api/projects/{project.id}/dataset/media?"
        f"annotation_status={DatasetItemAnnotationStatus.MISSING_ANNOTATIONS}"
    )
    actual_media_count = response.json()["pagination"]["total"]
    assert actual_media_count == count, f"Expected {count} unannotated media, got {actual_media_count}"
