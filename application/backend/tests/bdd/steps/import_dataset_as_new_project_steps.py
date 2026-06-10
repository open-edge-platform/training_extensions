# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import Any, cast
from uuid import UUID

from behave import given, then, when
from behave.runner import Context
from datumaro.experimental import Dataset
from datumaro.experimental.categories import Categories, LabelCategories
from requests import Session

from app.api.schemas import ProjectView
from app.api.schemas.jobs.dataset_import import ImportDatasetMetadata
from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.models import Task, TaskType
from tests.bdd.utils import import_dataset_as_new_project

SAMPLE_TYPES: dict[TaskType | str, Any] = {
    TaskType.DETECTION: DetectionImportExportSample,
    TaskType.CLASSIFICATION: MulticlassClassificationImportExportSample,
    "multilabel": MultilabelClassificationImportExportSample,
    TaskType.INSTANCE_SEGMENTATION: InstanceSegmentationImportExportSample,
}


@given("A {task_type} dataset with labels {labels} exists")  # pyrefly: ignore
@given("An {task_type} dataset with labels {labels} exists")  # pyrefly: ignore
def step_dataset_with_labels_exists(context: Context, task_type: str, labels: str) -> None:
    """Create a dataset of specified type with the specified labels."""
    labels_list = ast.literal_eval(labels)
    label_categories: Categories = LabelCategories(labels=labels_list)
    context.labels = label_categories
    if task_type not in SAMPLE_TYPES:
        raise ValueError(f"Unsupported task type '{task_type}' for dataset creation")
    context.dataset = Dataset(SAMPLE_TYPES[task_type], categories={"label": label_categories})
    context.task = Task(
        task_type=TaskType(task_type) if task_type != "multilabel" else TaskType.CLASSIFICATION,
        exclusive_labels=task_type == TaskType.CLASSIFICATION,
    )


@when(  # pyrefly: ignore
    'I import the dataset as a new {task_type} project with name "{project_name}" and labels {labels}'
)
def step_import_dataset_as_new_project(context: Context, task_type: str, project_name: str, labels: str) -> None:
    """Import the dataset as a new project with the specified name and labels."""
    staged_dataset_id = cast(UUID, context.staged_dataset_id)
    labels_list = ast.literal_eval(labels)

    job = import_dataset_as_new_project(
        session=cast(Session, context.session),
        base_url=cast(str, context.base_url),
        project_name=project_name,
        staged_dataset_id=str(staged_dataset_id),
        task_type=TaskType(task_type) if task_type != "multilabel" else TaskType.CLASSIFICATION,
        labels=labels_list,
        exclusive_labels=task_type == TaskType.CLASSIFICATION,
    )
    context.project_id = cast(ImportDatasetMetadata, job.metadata).project_id


@then('the project "{project_name}" is created with labels {labels}')  # pyrefly: ignore
def step_project_created_with_labels(context: Context, project_name: str, labels: str) -> None:
    """Verify that the project was created with the expected name and labels."""
    session = cast(Session, context.session)
    response = session.get(f"{context.base_url}/api/projects/{context.project_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    project = ProjectView.model_validate(response.json())
    context.project = project
    assert project.name == project_name, f"Expected project name '{project_name}', got '{project.name}'"

    expected_labels = set(ast.literal_eval(labels))
    actual_labels = {label.name for label in project.task.labels}
    assert expected_labels == actual_labels, f"Expected labels {expected_labels}, got {actual_labels}"
