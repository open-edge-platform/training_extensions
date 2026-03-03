# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from datumaro.experimental.data_formats.base import DataFormat
from loguru import logger

from app.execution import PrepareDataset
from app.models.jobs import PrepareDatasetForImportJobParams


@pytest.fixture
def fxt_staged_datasets_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "staged_datasets"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def fxt_prepare(
    fxt_staged_datasets_dir: Path,
) -> PrepareDataset:
    return PrepareDataset(
        staged_datasets_dir=fxt_staged_datasets_dir,
    )


class TestPrepareDataset:
    def test_check_archive_no_directory(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        with pytest.raises(ValueError, match="Dataset directory does not exist"):
            fxt_prepare.check_archive(staged_dataset_id=uuid4())

    def test_check_archive_no_zip(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        dataset_dir.mkdir()
        with pytest.raises(ValueError, match="Cannot find dataset zip archive"):
            fxt_prepare.check_archive(staged_dataset_id=UUID(dataset_dir.name))

    def test_check_archive_multiple_zips(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        dataset_dir.mkdir()
        (dataset_dir / "dataset-coco.zip").write_bytes(b"zip1")
        (dataset_dir / "dataset-coco-2.zip").write_bytes(b"zip2")
        log_messages = []
        handler_id = logger.add(log_messages.append, level="WARNING", format="{message}")

        archive_path = fxt_prepare.check_archive(staged_dataset_id=UUID(dataset_dir.name))

        logger.remove(handler_id)

        assert archive_path.name == "dataset-coco-2.zip"
        assert len(log_messages) == 1
        assert "Found more than one zip archive" in log_messages[0]

    def test_check_archive_valid_zip(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        dataset_dir.mkdir()
        (dataset_dir / "dataset-coco.zip").write_bytes(b"zip")

        archive_path = fxt_prepare.check_archive(staged_dataset_id=UUID(dataset_dir.name))

        assert archive_path.name == "dataset-coco.zip"

    def test_convert_archive_invalid_format(self, fxt_prepare: PrepareDataset) -> None:
        with pytest.raises(ValueError, match="Unknown dataset format: UNKNOWN"):
            fxt_prepare.convert_archive(Path("dataset-UNKNOWN.zip"))

    def test_convert_archive_coco(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-coco.zip"
        extract_dir = archive_path.with_suffix("")
        dataset = Mock()
        with (
            patch(
                "app.execution.dataset_import.prepare._extract_archive", return_value=extract_dir
            ) as mock_extract_archive,
            patch("app.execution.dataset_import.prepare.load_dataset", return_value=dataset) as mock_load_dataset,
            patch("app.execution.dataset_import.prepare.export_dataset") as mock_export_dataset,
        ):
            fxt_prepare.convert_archive(archive_path)

            mock_extract_archive.assert_called_once_with(archive_path)
            mock_load_dataset.assert_called_once_with(
                data_format=DataFormat.COCO,
                images_dir_path=str(extract_dir / "images"),
                annotations_path=str(extract_dir / "annotations.json"),
            )
            mock_export_dataset.assert_called_once_with(
                dataset, output_path=archive_path.parent / "dataset", as_zip=False
            )

    def test_convert_archive_yolo(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-yolo.zip"
        extract_dir = archive_path.with_suffix("")
        dataset = Mock()
        with (
            patch(
                "app.execution.dataset_import.prepare._extract_archive", return_value=extract_dir
            ) as mock_extract_archive,
            patch("app.execution.dataset_import.prepare.load_dataset", return_value=dataset) as mock_load_dataset,
            patch("app.execution.dataset_import.prepare.export_dataset") as mock_export_dataset,
        ):
            fxt_prepare.convert_archive(archive_path)

            mock_extract_archive.assert_called_once_with(archive_path)
            mock_load_dataset.assert_called_once_with(
                data_format=DataFormat.YOLO,
                root_dir=str(extract_dir),
            )
            mock_export_dataset.assert_called_once_with(
                dataset, output_path=archive_path.parent / "dataset", as_zip=False
            )

    def test_convert_archive_geti(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-geti.zip"
        dataset = Mock()
        with (
            patch("app.execution.dataset_import.prepare.import_dataset", return_value=dataset) as mock_import_dataset,
            patch("app.execution.dataset_import.prepare.export_dataset") as mock_export_dataset,
        ):
            fxt_prepare.convert_archive(archive_path)

            mock_import_dataset.assert_called_once_with(str(archive_path), extract_dir=archive_path.parent / "dataset")
            mock_export_dataset.assert_not_called()

    def test_convert_archive_datumaro_v1(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-datumaro_v1.zip"
        legacy_dataset = Mock()
        dataset = Mock()
        with (
            patch(
                "app.execution.dataset_import.prepare.Dataset.import_from", return_value=legacy_dataset
            ) as mock_import_from,
            patch(
                "app.execution.dataset_import.prepare.convert_from_legacy", return_value=dataset
            ) as mock_convert_from_legacy,
            patch("app.execution.dataset_import.prepare.export_dataset") as mock_export_dataset,
        ):
            fxt_prepare.convert_archive(archive_path)

            mock_import_from.assert_called_once_with(str(archive_path))
            mock_convert_from_legacy.assert_called_once_with(legacy_dataset)
            mock_export_dataset.assert_called_once_with(
                dataset, output_path=archive_path.parent / "dataset", as_zip=False
            )

    def test_cleanup(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-coco.zip"
        extract_dir = archive_path.with_suffix("")
        extract_dir.mkdir(parents=True)
        archive_path.write_bytes(b"zip")

        fxt_prepare.cleanup(archive_path)

        assert not archive_path.exists()
        assert not extract_dir.exists()

    def test_execute(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        archive_path = dataset_dir / "dataset-coco.zip"

        with (
            patch.object(fxt_prepare, "check_archive", return_value=archive_path) as mock_check_archive,
            patch.object(fxt_prepare, "convert_archive") as mock_convert_archive,
            patch.object(fxt_prepare, "cleanup") as mock_cleanup,
        ):
            fxt_prepare.execute(PrepareDatasetForImportJobParams(staged_dataset_id=UUID(dataset_dir.name)))

            mock_check_archive.assert_called_once_with(UUID(dataset_dir.name))
            mock_convert_archive.assert_called_once_with(archive_path)
            mock_cleanup.assert_called_once_with(archive_path)
