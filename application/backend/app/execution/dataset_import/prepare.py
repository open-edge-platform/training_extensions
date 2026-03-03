# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
import zipfile
from pathlib import Path
from uuid import UUID

from datumaro.components.dataset import Dataset
from datumaro.experimental.data_formats.base import DataFormat, load_dataset
from datumaro.experimental.export_import import export_dataset, import_dataset
from datumaro.experimental.legacy import convert_from_legacy
from loguru import logger

from app.execution.base import Execution, step
from app.models import DatasetFormat
from app.models.jobs import PrepareDatasetForImportJobParams


def _extract_archive(archive_path: Path) -> Path:
    """
    Extracts a zip archive to the same directory.

    Args:
        archive_path: Path to the zip archive.

    Raises:
        ValueError: If the archive cannot be extracted.
    """
    try:
        extract_to = archive_path.with_suffix("")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to
    except zipfile.BadZipFile as e:
        raise ValueError(f"Failed to extract archive {archive_path}: {e}")


class PrepareDataset(Execution[PrepareDatasetForImportJobParams]):
    """
    Execution implementation for preparing dataset archives for import into Geti format.

    This class handles the conversion of various dataset formats (COCO, YOLO, VOC, Datumaro V1)
    into the Geti format. It extracts archived datasets, converts them to the standardized format,
    and cleans up temporary files.

    The execution follows these steps:
    1. Validate and locate the dataset archive in the staged directory
    2. Convert the dataset to Geti format based on the detected format
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
    def convert_archive(self, archive_path: Path) -> None:
        parts = archive_path.stem.split("-")
        if len(parts) != 2:
            raise ValueError(f"Cannot infer the format from the name: {archive_path.name}.")
        _, dataset_format = parts
        match dataset_format:
            case DatasetFormat.COCO:
                extract_dir = _extract_archive(archive_path)
                dataset = load_dataset(
                    data_format=DataFormat.COCO,
                    images_dir_path=str(extract_dir / "images"),
                    annotations_path=str(extract_dir / "annotations.json"),
                )
            case DatasetFormat.YOLO:
                extract_dir = _extract_archive(archive_path)
                dataset = load_dataset(
                    data_format=DataFormat.YOLO,
                    root_dir=str(extract_dir),
                )
            case DatasetFormat.VOC:
                # todo: implement after datumaro VOC exporter is implemented:
                #  https://github.com/open-edge-platform/datumaro/issues/2003
                raise NotImplementedError("VOC import is not implemented yet")
            case DatasetFormat.GETI:
                dataset = import_dataset(str(archive_path), extract_dir=archive_path.parent / "dataset")
            case DatasetFormat.DATUMARO_V1:
                legacy_dataset = Dataset.import_from(str(archive_path))
                dataset = convert_from_legacy(legacy_dataset)
            case _:
                raise ValueError(f"Unknown dataset format: {dataset_format}")

        if dataset_format != DatasetFormat.GETI:
            export_dataset(dataset, output_path=archive_path.parent / "dataset", as_zip=False)

    @step("Clean up original archive", 100)
    def cleanup(self, archive_path: Path) -> None:
        if archive_path.with_suffix("").exists():
            shutil.rmtree(archive_path.with_suffix(""))
        archive_path.unlink()

    def execute(self, params: PrepareDatasetForImportJobParams) -> None:
        archive_path = self.check_archive(params.staged_dataset_id)
        self.convert_archive(archive_path)
        self.cleanup(archive_path)
