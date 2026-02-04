# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, call, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.base import DataFormat

from app.executors import DatasetExporter
from app.models import (
    DatasetFormat,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    ExportDatasetJobParams,
    Task,
    TaskType,
)
from app.services.datumaro_converter import convert_to_dm_subset


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
) -> DatasetExporter:
    return DatasetExporter(
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
        fxt_exporter: DatasetExporter,
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
        dataset.filter_by_subset.assert_has_calls(
            [call(subset=convert_to_dm_subset(subset)) for subset in (subsets or [])]
        )

    @pytest.mark.parametrize(
        "subsets", [[DatasetItemSubset.TESTING], [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION], None]
    )
    def test_prepare_dataset_revision(
        self,
        subsets: list[DatasetItemSubset],
        fxt_exporter: DatasetExporter,
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
        dataset.filter_by_subset.assert_has_calls(
            [call(subset=convert_to_dm_subset(subset)) for subset in (subsets or [])]
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
        fxt_exporter: DatasetExporter,
        fxt_staged_datasets_dir: Path,
    ):
        dataset = Mock(spec=Dataset)
        dataset_id = uuid4()

        with patch("app.executors.dataset_export.exporter.save_dataset") as mock_save_dataset:
            target_dir = fxt_exporter.export_dataset(dataset_id, dataset, export_format)

            assert target_dir
            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            if export_format == DatasetFormat.COCO:
                mock_save_dataset.assert_called_once_with(
                    dataset=dataset,
                    data_format=data_format,
                    images_dir_path=str(target_dir / "images"),
                    annotations_path=str(target_dir / "annotations.json"),
                )
            elif export_format == DatasetFormat.YOLO:
                mock_save_dataset.assert_called_once_with(
                    dataset=dataset,
                    data_format=data_format,
                    root_dir=str(target_dir),
                )

    def test_export_dataset_datumaro_v2(self, fxt_exporter: DatasetExporter, fxt_staged_datasets_dir: Path):
        dataset = Mock(spec=Dataset)
        dataset_id = uuid4()

        with patch("app.executors.dataset_export.exporter.export_dataset") as mock_export_dataset:
            target_dir = fxt_exporter.export_dataset(dataset_id, dataset, DatasetFormat.DATUMARO_V2)

            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            mock_export_dataset.assert_called_once_with(dataset=dataset, output_path=target_dir, as_zip=True)

    def test_zip_dataset_contents(self, fxt_exporter: DatasetExporter, fxt_staged_datasets_dir: Path):
        target_dir = fxt_staged_datasets_dir / str(uuid4())
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "file1.txt").write_text("content1")
        (target_dir / "file2.txt").write_text("content2")

        zipped_path = fxt_exporter.zip_dataset_contents(target_dir=target_dir, export_format=DatasetFormat.COCO)

        assert zipped_path == target_dir / "dataset-coco.zip"
        assert zipped_path.exists()

    def test_cleanup(self, fxt_exporter: DatasetExporter, fxt_staged_datasets_dir: Path):
        target_dir = fxt_staged_datasets_dir / str(uuid4())
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "file1.txt").write_text("content1")
        (target_dir / "file2.txt").write_text("content2")
        zip_path = target_dir / "dataset-coco.zip"
        zip_path.write_text("zip content")

        fxt_exporter.cleanup(zip_path=zip_path)

        assert (target_dir / "file1.txt").exists() is False
        assert (target_dir / "file2.txt").exists() is False
        assert zip_path.exists()
