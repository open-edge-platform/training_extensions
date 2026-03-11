# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset

from app.execution import ImportDatasetToProject
from app.models import Task, TaskType
from app.models.jobs import ImportDatasetToProjectJobParams


@pytest.fixture
def fxt_import(
    fxt_staged_datasets_dir: Path,
    fxt_dataset_service: Mock,
    fxt_label_service: Mock,
    fxt_media_service: Mock,
    fxt_db_session_factory: Callable,
) -> ImportDatasetToProject:
    return ImportDatasetToProject(
        staged_datasets_dir=fxt_staged_datasets_dir,
        dataset_service=fxt_dataset_service,
        label_service=fxt_label_service,
        media_service=fxt_media_service,
        db_session_factory=fxt_db_session_factory,
    )


@pytest.fixture
def fxt_import_params() -> ImportDatasetToProjectJobParams:
    return ImportDatasetToProjectJobParams(
        project_id=uuid4(),
        task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
        staged_dataset_id=uuid4(),
        labels_mapping=None,
    )


class TestImportDatasetToProject:
    def test_prepare_dataset(
        self, fxt_import: ImportDatasetToProject, fxt_import_params: ImportDatasetToProjectJobParams
    ) -> None:
        dataset = Mock(spec=Dataset)

        with patch.object(fxt_import, "_prepare_dataset", return_value=dataset) as mock_prepare:
            result = fxt_import.prepare_dataset(
                staged_dataset_id=fxt_import_params.staged_dataset_id, task=fxt_import_params.task
            )

            mock_prepare.assert_called_once_with(
                staged_dataset_id=fxt_import_params.staged_dataset_id, task=fxt_import_params.task
            )
            assert result == dataset

    def test_create_items(
        self, fxt_import: ImportDatasetToProject, fxt_import_params: ImportDatasetToProjectJobParams
    ) -> None:
        dataset = Mock(spec=Dataset)

        with patch.object(fxt_import, "_create_items") as mock_create:
            fxt_import.create_items(dataset=dataset, params=fxt_import_params)

            mock_create.assert_called_once_with(
                dataset=dataset,
                project_id=fxt_import_params.project_id,
                task=fxt_import_params.task,
                labels_mapping={},
                include_unannotated=True,
            )

    def test_execute(
        self, fxt_import: ImportDatasetToProject, fxt_import_params: ImportDatasetToProjectJobParams
    ) -> None:
        dataset = Mock(spec=Dataset)

        with (
            patch.object(fxt_import, "prepare_dataset", return_value=dataset) as mock_prepare,
            patch.object(fxt_import, "create_items") as mock_create,
        ):
            fxt_import.execute(fxt_import_params)

            mock_prepare.assert_called_once_with(
                staged_dataset_id=fxt_import_params.staged_dataset_id, task=fxt_import_params.task
            )
            mock_create.assert_called_once_with(dataset=dataset, params=fxt_import_params)
