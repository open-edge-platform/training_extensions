# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import zipfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID, uuid4

from datumaro.experimental.data_formats.base import DataFormat, Dataset, save_dataset
from datumaro.experimental.export_import import export_dataset
from loguru import logger
from sqlalchemy.orm import Session

from app.core.run import ExecutionContext
from app.executors.base import Executor, step
from app.models import DatasetFormat, DatasetItemAnnotationStatus, ExportDatasetJobParams
from app.services import DatasetRevisionService, DatasetService
from app.services.datumaro_converter import convert_to_dm_subset


class DatasetExporter(Executor):
    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        dataset_revision_service: DatasetRevisionService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ):
        super().__init__()
        self._staged_datasets_dir = staged_datasets_dir
        self._dataset_service = dataset_service
        self._dataset_revision_service = dataset_revision_service
        self._db_session_factory = db_session_factory

    @step("Prepare dataset for export", 10)
    def prepare_dataset(self, export_params: ExportDatasetJobParams) -> tuple[UUID, Dataset | None]:
        with self._db_session_factory() as session:
            if export_params.dataset_id is None:
                self._dataset_service.set_db_session(session)
                annotation_status = (
                    DatasetItemAnnotationStatus.REVIEWED_WITH_UNANNOTATED
                    if export_params.include_unannotated
                    else DatasetItemAnnotationStatus.REVIEWED
                )
                dataset = self._dataset_service.get_dm_dataset(
                    project_id=export_params.project_id,
                    task=export_params.task,
                    annotation_status=annotation_status,
                    label_names=export_params.labels,
                )
            else:
                self._dataset_revision_service.set_db_session(session)
                dataset = self._dataset_revision_service.load_revision(
                    project_id=export_params.project_id, dataset_revision_id=export_params.dataset_id
                )
            filtered_dataset = None
            if export_params.subsets:
                # todo: datumaro API seems to lack a proper way to filter by multiple subsets,
                #  so we chain multiple filter calls for now
                for subset in export_params.subsets:
                    if filtered_dataset:
                        filtered_dataset.append_dataset(dataset.filter_by_subset(subset=convert_to_dm_subset(subset)))
                    filtered_dataset = dataset.filter_by_subset(subset=convert_to_dm_subset(subset))
                dataset = filtered_dataset
            return export_params.dataset_id or uuid4(), dataset

    @step("Export dataset", 80)
    def export_dataset(self, dataset_id: UUID, dataset: Dataset, export_format: DatasetFormat) -> Path | None:
        target_dir = self._staged_datasets_dir / str(dataset_id)
        logger.info("Exporting dataset {} to {} in {} format", dataset_id, target_dir, export_format)
        target_dir.mkdir(parents=True, exist_ok=True)
        match export_format:
            case DatasetFormat.COCO:
                save_dataset(
                    dataset=dataset,
                    data_format=self.__get_dm_format(export_format),
                    images_dir_path=str(target_dir / "images"),
                    annotations_path=str(target_dir / "annotations.json"),
                )
            case DatasetFormat.YOLO:
                save_dataset(
                    dataset=dataset,
                    data_format=self.__get_dm_format(export_format),
                    root_dir=str(target_dir),
                )
            case DatasetFormat.DATUMARO_V2:
                export_dataset(
                    dataset=dataset,
                    output_path=target_dir,
                    as_zip=True,
                )
            case _:
                raise ValueError(f"Unsupported dataset format for export: {export_format}")
        return target_dir

    @step("Compress dataset folder to zip archive", 90)
    def zip_dataset_contents(self, target_dir: Path, export_format: str) -> Path:
        zip_name = f"dataset-{export_format}.zip"
        zip_path = target_dir / zip_name

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(target_dir):
                root_path = Path(root)
                for file_name in files:
                    file_path = root_path / file_name
                    if file_path == zip_path:
                        continue
                    arcname = file_path.relative_to(target_dir)
                    zf.write(file_path, arcname)
        return zip_path

    @step("Remove dataset folder contents", 100)
    def cleanup(self, zip_path: Path) -> None:
        for item in zip_path.parent.iterdir():
            if item == zip_path:
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def run(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        export_params = ExportDatasetJobParams.model_validate_json(ctx.payload)
        dataset_id, dataset = self.prepare_dataset(export_params)
        if not dataset:
            logger.warning(
                "Dataset {} for project {} is empty after applying filters",
                export_params.dataset_id,
                export_params.project_id,
            )
            return
        target_dir = self.export_dataset(dataset_id, dataset, export_params.export_format)
        if export_params.export_format != DatasetFormat.DATUMARO_V2:
            # todo: remove after https://github.com/open-edge-platform/datumaro/issues/2026 is implemented
            zip_path = self.zip_dataset_contents(target_dir, export_params.export_format)
            self.cleanup(zip_path)

    @staticmethod
    def __get_dm_format(dataset_format: DatasetFormat) -> DataFormat:
        format_mapping = {
            DatasetFormat.COCO: DataFormat.COCO,
            DatasetFormat.YOLO: DataFormat.YOLO,
        }
        if dataset_format not in format_mapping:
            raise ValueError(f"Unsupported dataset format for export: {dataset_format}")
        return format_mapping[dataset_format]
