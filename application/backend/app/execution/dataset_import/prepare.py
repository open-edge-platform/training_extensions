# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from uuid import UUID

from datumaro.experimental.export_import import export_dataset, import_dataset
from loguru import logger

from app.execution.base import Execution, step
from app.models.jobs import PrepareDatasetForImportJobParams


class PrepareDataset(Execution[PrepareDatasetForImportJobParams]):
    """
    Execution implementation for preparing dataset archives for import into Geti format.

    This class handles the conversion of various dataset formats (COCO, YOLO, VOC, Datumaro V1)
    into the Geti format. It extracts archived datasets, converts them to the standardized format,
    and cleans up temporary files.

    The execution follows these steps:
    1. Validate and locate the dataset archive in the staged directory
    2. Convert the dataset to Geti format using Datumaro's import/export functionality
    3. Clean up the original archive and extracted files

    Supported formats:
    - COCO: Images and annotations in COCO JSON format
    - YOLO: YOLO format with root directory structure
    - VOC: Pascal VOC format (not yet implemented)
    - GETI: Native Geti format (pass-through)
    - DATUMARO_V1: Legacy Datumaro v1 format

    Attributes:
        params_type: The parameter type for this execution (PrepareDatasetForImportJobParams).

    Args:
        staged_datasets_dir: Path to the directory containing staged dataset archives.

    Raises:
        ValueError: If the dataset archive is not found, cannot be extracted, or has an invalid format.
        NotImplementedError: If attempting to import VOC format (not yet supported).
    """

    params_type = PrepareDatasetForImportJobParams

    def __init__(self, staged_datasets_dir: Path) -> None:
        super().__init__()
        self._staged_datasets_dir = staged_datasets_dir

    @step("Check dataset archive", 5)
    def check_archive(self, staged_dataset_id: UUID) -> Path:
        dataset_dir = self._staged_datasets_dir / str(staged_dataset_id)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
        zip_archives = list(dataset_dir.glob("*.zip"))
        if not zip_archives:
            raise ValueError(f"Cannot find dataset zip archive in {dataset_dir}")
        dataset_archive = sorted(zip_archives)[0]
        if len(zip_archives) > 1:
            logger.warning(f"Found more than one zip archive in {dataset_dir}. Using the first one: {dataset_archive}")
        return dataset_archive

    @step("Convert dataset archive to Geti format", 90)
    def convert_archive(self, archive_path: Path) -> Path:
        tmp_dir = archive_path.parent / f"{archive_path.stem}_import"
        dataset = import_dataset(archive_path, extract_dir=tmp_dir)
        export_dataset(dataset, output_path=archive_path.parent / "dataset", as_zip=False)
        return tmp_dir

    @step("Clean up original archive", 100)
    def cleanup(self, archive_path: Path, tmp_dir: Path) -> None:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        archive_path.unlink()

    def execute(self, params: PrepareDatasetForImportJobParams) -> None:
        archive_path = self.check_archive(params.staged_dataset_id)
        tmp_dir = self.convert_archive(archive_path)
        self.cleanup(archive_path, tmp_dir)
