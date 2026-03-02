# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import zipfile
from pathlib import Path

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.base import load_dataset
from datumaro.experimental.export_import import import_dataset

from app.execution.dataset_export import get_dm_format
from app.models import DatasetFormat


def import_dataset_by_format(dataset_path: Path, dataset_format: DatasetFormat) -> Dataset:
    match dataset_format:
        case DatasetFormat.GETI:
            return import_dataset(dataset_path)
        case DatasetFormat.YOLO:
            extract_dir = dataset_path.with_suffix("")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            return load_dataset(
                data_format=get_dm_format(dataset_format),
                root_dir=str(extract_dir),
            )
        case DatasetFormat.COCO:
            extract_dir = dataset_path.with_suffix("")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            return load_dataset(
                data_format=get_dm_format(dataset_format),
                images_dir_path=str(extract_dir / "images"),
                annotations_path=str(extract_dir / "annotations.json"),
            )
        case _:
            raise Exception(f"Unknown format: {dataset_format}")
