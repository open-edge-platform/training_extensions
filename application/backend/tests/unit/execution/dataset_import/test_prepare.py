# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from datumaro.experimental.categories import HierarchicalLabelCategories
from loguru import logger

from app.execution import PrepareDataset
from app.execution.base import ExecutionErr
from app.models.jobs import PrepareDatasetForImportJobParams


@pytest.fixture
def fxt_prepare(
    fxt_staged_datasets_dir: Path,
) -> PrepareDataset:
    return PrepareDataset(
        staged_datasets_dir=fxt_staged_datasets_dir,
    )


class TestPrepareDataset:
    def test_check_archive_no_directory(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        with pytest.raises(ExecutionErr, match="Dataset directory does not exist"):
            fxt_prepare.check_archive(staged_dataset_id=uuid4())

    def test_check_archive_no_zip(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        dataset_dir.mkdir()
        with pytest.raises(ExecutionErr, match="Cannot find dataset zip archive"):
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

    def test_convert_archive_dataset_error(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset.zip"
        with (
            pytest.raises(
                ExecutionErr,
                match="The dataset could not be recognized in any of the supported formats. "
                "Please verify that the dataset is well-formed and in a supported "
                "format; if the problem persists, report the issue.",
            ),
            patch("app.execution.dataset_import.prepare.import_dataset", side_effect=ValueError),
        ):
            fxt_prepare.convert_archive(archive_path)

    def test_convert_archive_hierarchical_error(
        self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path
    ) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset.zip"
        archive_path.parent.mkdir(parents=True)
        dataset = Mock()
        dataset.label_categories = HierarchicalLabelCategories()
        with (
            pytest.raises(
                ExecutionErr,
                match="The dataset with hierarchical labels is not supported.",
            ),
            patch("app.execution.dataset_import.prepare.import_dataset", return_value=dataset),
        ):
            fxt_prepare.convert_archive(archive_path)

        assert not archive_path.parent.exists()

    def test_convert_archive_success(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset.zip"
        dataset = Mock()
        with (
            patch("app.execution.dataset_import.prepare.import_dataset", return_value=dataset) as mock_import_dataset,
            patch("app.execution.dataset_import.prepare.export_dataset") as mock_export_dataset,
        ):
            fxt_prepare.convert_archive(archive_path)

            mock_import_dataset.assert_called_once_with(
                archive_path, extract_dir=archive_path.parent / f"{archive_path.stem}_import"
            )
            mock_export_dataset.assert_called_once_with(
                dataset, output_path=archive_path.parent / "dataset", as_zip=False
            )

    def test_cleanup(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        archive_path = fxt_staged_datasets_dir / str(uuid4()) / "dataset-coco.zip"
        tmp_dir = Path(f"{archive_path.with_suffix('')}_import")
        tmp_dir.mkdir(parents=True)
        archive_path.write_bytes(b"zip")

        fxt_prepare.cleanup(archive_path, tmp_dir)

        assert not archive_path.exists()
        assert not tmp_dir.exists()

    def test_execute(self, fxt_prepare: PrepareDataset, fxt_staged_datasets_dir: Path) -> None:
        dataset_dir = fxt_staged_datasets_dir / str(uuid4())
        archive_path = dataset_dir / "dataset-coco.zip"
        tmp_dir = archive_path.parent / f"{archive_path.stem}_import"

        with (
            patch.object(fxt_prepare, "check_archive", return_value=archive_path) as mock_check_archive,
            patch.object(fxt_prepare, "convert_archive", return_value=tmp_dir) as mock_convert_archive,
            patch.object(fxt_prepare, "cleanup") as mock_cleanup,
        ):
            fxt_prepare.execute(PrepareDatasetForImportJobParams(staged_dataset_id=UUID(dataset_dir.name)))

            mock_check_archive.assert_called_once_with(UUID(dataset_dir.name))
            mock_convert_archive.assert_called_once_with(archive_path)
            mock_cleanup.assert_called_once_with(archive_path, tmp_dir)
