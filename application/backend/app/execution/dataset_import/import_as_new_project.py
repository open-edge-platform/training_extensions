# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import uuid4

from datumaro.experimental import Dataset
from datumaro.experimental.fields import Subset
from sqlalchemy.orm import Session

from app.execution.base import step
from app.models import Label, Project, Task
from app.models.jobs import ImportDatasetAsNewProjectJobParams
from app.services import DatasetService, LabelService, MediaService, ProjectService

from .base_import import BaseDatasetImport


class ImportDatasetAsNewProject(BaseDatasetImport[ImportDatasetAsNewProjectJobParams]):
    """
    Execution implementation for importing datasets as new Geti projects.

    This class handles the full workflow of creating a new project from a staged dataset,
    including project creation with labels, dataset preparation, and item import.

    The execution follows these steps:
    1. Create a new project with the specified task type and labels
    2. Load and filter the prepared dataset from the staged directory
    3. Create media items and dataset items with annotations for each dataset entry

    Attributes:
        params_type: The parameter type for this execution (ImportDatasetAsNewProjectJobParams).

    Args:
        staged_datasets_dir: Path to the directory containing staged dataset files.
        project_service: Service for managing project creation and operations.
        dataset_service: Service for managing dataset items and operations.
        label_service: Service for managing project labels.
        media_service: Service for managing media items (images).
        db_session_factory: Factory for creating database sessions.
    """

    params_type = ImportDatasetAsNewProjectJobParams

    def __init__(
        self,
        staged_datasets_dir: Path,
        project_service: ProjectService,
        dataset_service: DatasetService,
        label_service: LabelService,
        media_service: MediaService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        super().__init__(staged_datasets_dir, dataset_service, label_service, media_service, db_session_factory)
        self._project_service = project_service

    @step("Create new project", 5)
    def create_project(self, params: ImportDatasetAsNewProjectJobParams) -> Project:
        project_labels = (
            [
                Label(
                    id=uuid4(),
                    name=label_name,
                    color=f"#{secrets.token_hex(3).upper()}",
                )
                for label_name in params.labels
            ]
            if params.labels
            else []
        )
        task = Task(
            task_type=params.task_type,
            labels=project_labels,
            exclusive_labels=params.exclusive_labels,
        )
        with self._db_session_factory() as db_session:
            self._project_service.set_db_session(db_session)
            return self._project_service.create_project(project_id=uuid4(), name=params.project_name, task=task)

    @step("Prepare dataset", 15)
    def prepare_dataset(self, params: ImportDatasetAsNewProjectJobParams, task: Task) -> Dataset:
        dataset = self._prepare_dataset(staged_dataset_id=params.staged_dataset_id, task=task)
        if len(dataset) > 0 and params.subsets:
            dataset = dataset.filter_by_subset(subset=[Subset[subset.name] for subset in params.subsets])
        if len(dataset) > 0 and params.labels:
            dataset = dataset.filter_by_labels(labels=params.labels, keep_empty_samples=params.include_unannotated)
        return dataset

    @step("Import items from dataset to project", 100)
    def create_items(self, dataset: Dataset, project: Project, include_unannotated: bool) -> None:
        cats = self._get_dataset_label_categories(dataset)
        project_labels = [label.name for label in project.task.labels]
        # Labels that are in the dataset but not in the project will be mapped to None,
        # which means that their annotations will be imported or not without a label association.
        labels_mapping: dict[str, str | None] = {
            label_name: None for label_name in cats.labels if label_name not in project_labels
        }
        return self._create_items(
            dataset=dataset,
            project_id=project.id,
            task=project.task,
            labels_mapping=labels_mapping,
            include_unannotated=include_unannotated,
            start_progress=15.0,
        )

    def execute(self, params: ImportDatasetAsNewProjectJobParams) -> None:
        project = self.create_project(params)
        self.update_metadata({"project_id": project.id})
        dataset = self.prepare_dataset(params=params, task=project.task)
        self.create_items(dataset=dataset, project=project, include_unannotated=params.include_unannotated)
