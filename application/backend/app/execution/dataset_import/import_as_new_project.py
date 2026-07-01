# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
import types
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Union, get_args, get_origin
from uuid import uuid4

import numpy as np
import polars as pl
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
    1. Load dataset from the staged directory
    2. Create a new project with the specified task type and labels
    3. Convert and filter the loaded dataset according to the project task and import parameters
    4. Create media items and dataset items with annotations for each dataset entry

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

    @step("Import dataset", 5)
    def import_dataset(self, params: ImportDatasetAsNewProjectJobParams) -> Dataset:
        return self._import_dataset(staged_dataset_id=params.staged_dataset_id)

    @step("Create new project", 10)
    def create_project(self, params: ImportDatasetAsNewProjectJobParams, exclusive_labels: bool) -> Project:
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
        task = Task(task_type=params.task_type, labels=project_labels, exclusive_labels=exclusive_labels)
        with self._db_session_factory() as db_session:
            self._project_service.set_db_session(db_session)
            return self._project_service.create_project(project_id=uuid4(), name=params.project_name, task=task)

    @step("Prepare dataset", 15)
    def prepare_dataset(self, dataset: Dataset, params: ImportDatasetAsNewProjectJobParams, task: Task) -> Dataset:
        dataset = self._convert_dataset(dataset=dataset, task=task)
        if len(dataset) > 0 and params.subsets:
            dataset = dataset.filter_by_subset(subset=[Subset[subset.name] for subset in params.subsets])
        if len(dataset) > 0 and params.labels:
            # Track items that were explicitly labeled as empty (i.e., reviewed but intentionally have no labels)
            # BEFORE filtering. This is necessary because filter_by_labels with keep_empty_samples=True will also keep
            # genuinely unannotated items, making them indistinguishable from items that originally had empty labels.
            # After filtering, we reset user_reviewed=False for any newly-empty items (those that lost their labels
            # due to filtering) while preserving user_reviewed=True for items that were already empty-labeled before
            # the filter.
            empty_label_supported = "user_reviewed" in dataset.df.columns and dataset.df["label"].dtype == pl.List
            empty_labeled = set()
            if empty_label_supported:
                mask = (pl.col("label").list.len() == 0) & pl.col("user_reviewed")
                empty_labeled = set(dataset.df.filter(mask)["id"].to_list())
            dataset = dataset.filter_by_labels(labels=params.labels, keep_empty_samples=params.include_unannotated)
            if empty_label_supported:
                dataset.df = dataset.df.with_columns(
                    pl.when((pl.col("label").list.len() == 0) & (~pl.col("id").is_in(empty_labeled)))
                    .then(False)
                    .otherwise(pl.col("user_reviewed"))
                    .alias("user_reviewed")
                )
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
        dataset = self.import_dataset(params=params)
        has_array_label = self._has_array_label(dataset=dataset)
        project = self.create_project(params=params, exclusive_labels=not has_array_label)
        self.update_metadata({"project_id": project.id})
        dataset = self.prepare_dataset(dataset=dataset, params=params, task=project.task)
        self.create_items(dataset=dataset, project=project, include_unannotated=params.include_unannotated)

    @staticmethod
    def _has_array_label(dataset: Dataset) -> bool:
        """
        Check if the dataset's 'label' attribute is represented as a NumPy array.

        This method inspects the schema definition of the 'label' attribute to check if its type is or contains
        `np.ndarray` (such as `np.ndarray | None`).

        The result impacts how the project is created: if the dataset has array labels, the newly created project will
        configure its task with non-exclusive labels (i.e., a multi-label classification project).

        Args:
            dataset: The Datumaro dataset to analyze.

        Returns:
            True if the dataset's 'label' attribute schema type contains `np.ndarray`,
            False otherwise.
        """
        if "label" not in dataset.schema.attributes:
            return False

        label_type = dataset.schema.attributes["label"].type

        def _is_numpy_array_type(tp: object) -> bool:
            return tp is np.ndarray or get_origin(tp) is np.ndarray

        origin = get_origin(label_type)
        if origin in (Union, types.UnionType):
            return any(_is_numpy_array_type(arg) for arg in get_args(label_type))

        return _is_numpy_array_type(label_type)
