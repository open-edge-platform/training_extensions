# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID

from datumaro.experimental import Dataset
from sqlalchemy.orm import Session

from app.execution.base import step
from app.models import Task
from app.models.jobs import ImportDatasetToProjectJobParams
from app.services import DatasetService, LabelService, MediaService

from .base_import import BaseDatasetImport


class ImportDatasetToProject(BaseDatasetImport[ImportDatasetToProjectJobParams]):
    """
    Execution implementation for importing datasets into Geti projects.

    This class handles the import of previously prepared datasets in Geti format into a project.
    It loads the dataset from the staged directory, creates media items, and populates the project
    with dataset items including their annotations and metadata.

    The execution follows these steps:
    1. Load the prepared dataset from the staged directory
    2. Retrieve project labels for annotation mapping
    3. Create media items and dataset items with annotations for each dataset entry

    Attributes:
        params_type: The parameter type for this execution (ImportDatasetToProjectJobParams).

    Args:
        staged_datasets_dir: Path to the directory containing staged dataset files.
        dataset_service: Service for managing dataset items and operations.
        label_service: Service for managing project labels.
        media_service: Service for managing media items (images).
        db_session_factory: Factory for creating database sessions.
    """

    params_type = ImportDatasetToProjectJobParams

    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        label_service: LabelService,
        media_service: MediaService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        super().__init__(staged_datasets_dir, dataset_service, label_service, media_service, db_session_factory)

    @step("Prepare dataset", 10)
    def prepare_dataset(self, staged_dataset_id: UUID, task: Task) -> Dataset:
        return self._prepare_dataset(staged_dataset_id=staged_dataset_id, task=task)

    @step("Import items from dataset to project", 100)
    def create_items(self, dataset: Dataset, params: ImportDatasetToProjectJobParams) -> None:
        return self._create_items(
            dataset=dataset,
            project_id=params.project_id,
            task=params.task,
            labels_mapping=params.labels_mapping or {},
            include_unannotated=params.include_unannotated,
        )

    def execute(self, params: ImportDatasetToProjectJobParams) -> None:
        dataset = self.prepare_dataset(staged_dataset_id=params.staged_dataset_id, task=params.task)
        self.create_items(dataset=dataset, params=params)
