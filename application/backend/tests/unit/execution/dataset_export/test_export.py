# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.base import DataFormat

from app.datumaro_converter.utils import SubsetConverter
from app.execution import DatasetExport
from app.models import (
    DatasetFormat,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    ExportDatasetJobParams,
    Task,
    TaskType,
)


@pytest.fixture
def fxt_staged_datasets_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "staged_datasets"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def fxt_exporter(
    fxt_staged_datasets_dir: Path,
    fxt_dataset_service: Mock,
    fxt_dataset_revision_service: Mock,
    fxt_db_session_factory: Callable,
) -> DatasetExport:
    return DatasetExport(
        staged_datasets_dir=fxt_staged_datasets_dir,
        dataset_service=fxt_dataset_service,
        dataset_revision_service=fxt_dataset_revision_service,
        db_session_factory=fxt_db_session_factory,
    )


@pytest.fixture
def fxt_export_params() -> ExportDatasetJobParams:
    return ExportDatasetJobParams(
        project_id=uuid4(),
        task=Task(task_type=TaskType.DETECTION),
        export_format=DatasetFormat.COCO,
        labels=["label1", "label2"],
        subsets=[DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION],
        include_unannotated=False,
    )


class TestDatasetExporter:
    @pytest.mark.parametrize(
        "subsets", [[DatasetItemSubset.TESTING], [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION], None]
    )
    def test_prepare_dataset_project(
        self,
        subsets: list[DatasetItemSubset],
        fxt_exporter: DatasetExport,
        fxt_dataset_service: Mock,
        fxt_export_params: ExportDatasetJobParams,
    ):
        dataset = Mock(spec=Dataset)
        fxt_dataset_service.get_dm_dataset.return_value = dataset
        fxt_export_params.subsets = subsets

        fxt_exporter.prepare_dataset(fxt_export_params)

        fxt_dataset_service.get_dm_dataset.assert_called_once_with(
            project_id=fxt_export_params.project_id,
            task=fxt_export_params.task,
            annotation_status=DatasetItemAnnotationStatus.REVIEWED,
            label_names=fxt_export_params.labels,
        )
        if subsets:
            dataset.filter_by_subset.assert_called_once_with(
                subset=[SubsetConverter.to_datumaro(subset) for subset in subsets]
            )

    @pytest.mark.parametrize(
        "subsets", [[DatasetItemSubset.TESTING], [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION], None]
    )
    def test_prepare_dataset_revision(
        self,
        subsets: list[DatasetItemSubset],
        fxt_exporter: DatasetExport,
        fxt_dataset_revision_service: Mock,
        fxt_export_params: ExportDatasetJobParams,
    ):
        fxt_export_params.dataset_id = uuid4()
        fxt_export_params.subsets = subsets
        dataset = Mock(spec=Dataset)
        fxt_dataset_revision_service.load_revision.return_value = dataset

        dataset_id, _ = fxt_exporter.prepare_dataset(fxt_export_params)

        assert dataset_id == fxt_export_params.dataset_id
        fxt_dataset_revision_service.load_revision.assert_called_once_with(
            project_id=fxt_export_params.project_id,
            dataset_revision_id=fxt_export_params.dataset_id,
        )
        if subsets:
            dataset.filter_by_subset.assert_called_once_with(
                subset=[SubsetConverter.to_datumaro(subset) for subset in subsets]
            )

    @pytest.mark.parametrize(
        "export_format, data_format",
        [
            (DatasetFormat.COCO, DataFormat.COCO),
            (DatasetFormat.YOLO, DataFormat.YOLO),
        ],
    )
    def test_export_dataset(
        self,
        export_format: DatasetFormat,
        data_format: DataFormat,
        fxt_exporter: DatasetExport,
        fxt_staged_datasets_dir: Path,
    ):
        dataset = Mock(spec=Dataset)
        dataset_id = uuid4()

        with patch("app.execution.dataset_export.export.save_dataset") as mock_save_dataset:
            target_dir = fxt_exporter.export_dataset(dataset_id, dataset, export_format)

            assert target_dir
            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            mock_save_dataset.assert_called_once_with(
                dataset=dataset,
                data_format=data_format,
                output_path=str(fxt_staged_datasets_dir / str(dataset_id) / f"dataset-{export_format}.zip"),
                as_zip=True,
            )

    def test_export_dataset_geti(self, fxt_exporter: DatasetExport, fxt_staged_datasets_dir: Path):
        dataset = Mock(spec=Dataset)
        dataset_id = uuid4()

        with (
            patch("app.execution.dataset_export.export.export_dataset") as mock_export_dataset,
        ):
            target_dir = fxt_exporter.export_dataset(dataset_id, dataset, DatasetFormat.GETI)

            assert target_dir
            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            mock_export_dataset.assert_called_once_with(
                dataset=dataset, output_path=str(target_dir / f"dataset-{DatasetFormat.GETI}.zip"), as_zip=True
            )
