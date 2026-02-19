# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID, uuid4

from datumaro.experimental.data_formats.base import DataFormat, Dataset, save_dataset
from datumaro.experimental.export_import import export_dataset
from loguru import logger
from sqlalchemy.orm import Session

from app.core.run import ExecutionContext
from app.datumaro_converter.utils import SubsetConverter
from app.execution.base import Execution, step
from app.models import DatasetFormat, DatasetItemAnnotationStatus, ExportDatasetJobParams
from app.services import DatasetRevisionService, DatasetService


def get_dm_format(dataset_format: DatasetFormat) -> DataFormat:
    """
    Convert DatasetFormat to Datumaro DataFormat.

    Args:
        dataset_format: DatasetFormat enum value.

    Returns:
        Corresponding DataFormat enum value.
    """
    format_mapping = {
        DatasetFormat.COCO: DataFormat.COCO,
        DatasetFormat.YOLO: DataFormat.YOLO,
    }
    if dataset_format not in format_mapping:
        raise ValueError(f"Unsupported dataset format for export: {dataset_format}")
    return format_mapping[dataset_format]


class ExportDataset(Execution):
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

    @step("Prepare dataset for export", 20)
    def prepare_dataset(self, export_params: ExportDatasetJobParams) -> tuple[UUID, Dataset | None]:
        with self._db_session_factory() as session:
            if export_params.dataset_id is None:
                self._dataset_service.set_db_session(session)
                annotation_status = (
                    DatasetItemAnnotationStatus.REVIEWED_OR_UNANNOTATED
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
            if dataset and export_params.subsets:
                dataset = dataset.filter_by_subset(
                    subset=[SubsetConverter.to_datumaro(subset) for subset in export_params.subsets]
                )
            return uuid4(), dataset

    @step("Export dataset", 100)
    def export_dataset(self, dataset_id: UUID, dataset: Dataset, export_format: DatasetFormat) -> Path | None:
        target_dir = self._staged_datasets_dir / str(dataset_id)
        logger.info("Exporting dataset {} to {} in {} format", dataset_id, target_dir, export_format)
        target_dir.mkdir(parents=True, exist_ok=True)
        match export_format:
            case DatasetFormat.COCO | DatasetFormat.YOLO:
                save_dataset(
                    dataset=dataset,
                    data_format=get_dm_format(export_format),
                    output_path=str(target_dir / f"dataset-{export_format}.zip"),
                    as_zip=True,
                )
            case DatasetFormat.VOC:
                # todo: implement after datumaro VOC exporter is implemented:
                #  https://github.com/open-edge-platform/datumaro/issues/2003
                raise NotImplementedError("VOC export is not implemented yet")
            case DatasetFormat.GETI:
                export_dataset(
                    dataset=dataset,
                    output_path=str(target_dir / f"dataset-{export_format}.zip"),
                    as_zip=True,
                )
            case _:
                raise ValueError(f"Unsupported dataset format for export: {export_format}")
        return target_dir

    def run(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        export_params = ExportDatasetJobParams.model_validate_json(ctx.payload)
        dataset_id, dataset = self.prepare_dataset(export_params)
        if not dataset:
            logger.warning(
                "Dataset {} for project {} is empty after applying filters. Nothing to export.",
                dataset_id,
                export_params.project_id,
            )
            return
        self.report_progress(metadata={"dataset_id": dataset_id})
        self.export_dataset(dataset_id, dataset, export_params.export_format)
