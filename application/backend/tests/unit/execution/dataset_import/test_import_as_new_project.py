# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories

from app.execution import ImportDatasetAsNewProject
from app.models import Label, Project, Task, TaskType
from app.models.jobs import ImportDatasetAsNewProjectJobParams
from app.services import ProjectService


@pytest.fixture
def fxt_project_service() -> Mock:
    """Mock ProjectService for testing."""
    return Mock(spec=ProjectService)


@pytest.fixture
def fxt_import(
    fxt_staged_datasets_dir: Path,
    fxt_project_service: Mock,
    fxt_dataset_service: Mock,
    fxt_label_service: Mock,
    fxt_media_service: Mock,
    fxt_db_session_factory: Callable,
) -> ImportDatasetAsNewProject:
    return ImportDatasetAsNewProject(
        staged_datasets_dir=fxt_staged_datasets_dir,
        project_service=fxt_project_service,
        dataset_service=fxt_dataset_service,
        label_service=fxt_label_service,
        media_service=fxt_media_service,
        db_session_factory=fxt_db_session_factory,
    )


@pytest.fixture
def fxt_import_params() -> ImportDatasetAsNewProjectJobParams:
    return ImportDatasetAsNewProjectJobParams(
        project_name="New Project",
        task_type=TaskType.CLASSIFICATION,
        exclusive_labels=False,
        staged_dataset_id=uuid4(),
        labels=["label1", "label2"],
        subsets=[],
    )


class TestImportDatasetToProject:
    @pytest.mark.parametrize("labels", [["label1", "label2"], None])
    def test_create_project(
        self,
        labels: list[str] | None,
        fxt_import: ImportDatasetAsNewProject,
        fxt_import_params: ImportDatasetAsNewProjectJobParams,
        fxt_project_service: Mock,
    ) -> None:
        fxt_import_params.labels = labels
        project = Mock(spec=Project)
        project.id = uuid4()
        fxt_project_service.create_project.return_value = project

        result = fxt_import.create_project(fxt_import_params)

        fxt_project_service.create_project.assert_called_once()
        call_args = fxt_project_service.create_project.call_args
        assert call_args.kwargs["name"] == fxt_import_params.project_name
        assert call_args.kwargs["task"].exclusive_labels == fxt_import_params.exclusive_labels
        assert call_args.kwargs["task"].task_type == fxt_import_params.task_type
        assert result == project

    def test_prepare_dataset(
        self, fxt_import: ImportDatasetAsNewProject, fxt_import_params: ImportDatasetAsNewProjectJobParams
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__len__.return_value = 10
        dataset.filter_by_subset.return_value = dataset
        dataset.filter_by_labels.return_value = dataset
        task = Task(task_type=fxt_import_params.task_type)

        with patch.object(fxt_import, "_prepare_dataset", return_value=dataset) as mock_prepare:
            result = fxt_import.prepare_dataset(params=fxt_import_params, task=task)

            mock_prepare.assert_called_once_with(staged_dataset_id=fxt_import_params.staged_dataset_id, task=task)
            dataset.filter_by_subset.assert_not_called()
            dataset.filter_by_labels.assert_called_once_with(
                labels=fxt_import_params.labels, keep_empty_samples=fxt_import_params.include_unannotated
            )
            assert result == dataset

    @pytest.mark.parametrize(
        "dataset_labels, project_labels, expected_mapping",
        [
            (
                ("label1", "label2", "label3"),
                ("label1", "label2"),
                {"label3": None},
            ),  # label3 is in dataset but not in project
            (("label1", "label2"), ("label1", "label2"), {}),  # all labels in dataset are in project
            (
                ("label1", "label2"),
                (),
                {"label1": None, "label2": None},
            ),  # project with empty labels, all dataset labels should be mapped to None
        ],
    )
    def test_create_items(
        self,
        dataset_labels: tuple[str, ...],
        project_labels: tuple[str, ...],
        expected_mapping: dict[str, str | None],
        fxt_import: ImportDatasetAsNewProject,
        fxt_import_params: ImportDatasetAsNewProjectJobParams,
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Task(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=False,
            labels=[
                Label(id=uuid4(), name=label_name, color=f"#{secrets.token_hex(3).upper()}", hotkey=None)
                for label_name in project_labels
            ],
        )

        with (
            patch.object(fxt_import, "_create_items") as mock_create,
            patch.object(
                fxt_import,
                "_get_dataset_label_categories",
                return_value=LabelCategories(
                    labels=dataset_labels,
                ),
            ),
        ):
            fxt_import.create_items(dataset=dataset, project=project, include_unannotated=False)

            mock_create.assert_called_once_with(
                dataset=dataset,
                project_id=project.id,
                task=project.task,
                labels_mapping=expected_mapping,
                include_unannotated=False,
            )

    def test_execute(
        self, fxt_import: ImportDatasetAsNewProject, fxt_import_params: ImportDatasetAsNewProjectJobParams
    ) -> None:
        dataset = Mock(spec=Dataset)
        project = Mock(spec=Project)
        project.id = uuid4()
        project.name = fxt_import_params.project_name
        project.task = Mock(spec=Task)
        with (
            patch.object(fxt_import, "create_project", return_value=project) as mock_create_project,
            patch.object(fxt_import, "prepare_dataset", return_value=dataset) as mock_prepare,
            patch.object(fxt_import, "create_items") as mock_create,
        ):
            fxt_import.execute(fxt_import_params)

            mock_create_project.assert_called_once_with(fxt_import_params)
            mock_prepare.assert_called_once_with(params=fxt_import_params, task=project.task)
            mock_create.assert_called_once_with(
                dataset=dataset, project=project, include_unannotated=fxt_import_params.include_unannotated
            )
