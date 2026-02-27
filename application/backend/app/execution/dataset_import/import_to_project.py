# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast
from uuid import UUID

from datumaro.experimental import Dataset, Sample
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import import_dataset
from loguru import logger
from sqlalchemy.orm import Session

from app.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)
from app.execution.base import Execution, step
from app.models import DatasetItemSubset, Task, TaskType
from app.models.jobs import ImportDatasetToProjectJobParams
from app.models.media import ImageFormat
from app.services import DatasetService, LabelService, MediaService

from .sample_to_annotation import DatumaroSampleToGetiAnnotationConverter


class ImportDatasetToProject(Execution[ImportDatasetToProjectJobParams]):
    """
    Execution implementation for importing datasets into Geti projects.

    This class handles the import of previously prepared datasets in Geti format into a project.
    It loads the dataset from the staged directory, creates media items, and populates the project
    with dataset items including their annotations and metadata.

    The execution follows these steps:
    1. Load the prepared dataset from the staged directory
    2. Retrieve project labels for annotation mapping
    3. Create media items and dataset items with annotations for each dataset entry

    **Note**: Items are currently created one by one in sequential order. This approach may impact
    performance for large datasets and could be optimized by implementing batch processing to reduce
    database overhead and improve throughput.

    Progress is reported in increments of 5% during the import process.

    Attributes:
        params_type: The parameter type for this execution (ImportDatasetToProjectJobParams).
        BATCH_PROGRESS_INTERVAL: Number of batches for progress reporting (20 batches = 5% intervals).

    Args:
        staged_datasets_dir: Path to the directory containing staged dataset files.
        dataset_service: Service for managing dataset items and operations.
        label_service: Service for managing project labels.
        media_service: Service for managing media items (images).
        db_session_factory: Factory for creating database sessions.

    Raises:
        ValueError: If the staged dataset directory does not exist or dataset import fails.
    """

    params_type = ImportDatasetToProjectJobParams
    BATCH_PROGRESS_INTERVAL = 20  # 5% intervals (100% / 5% = 20)

    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        label_service: LabelService,
        media_service: MediaService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        super().__init__()
        self._staged_datasets_dir = staged_datasets_dir
        self._media_service = media_service
        self._label_service = label_service
        self._dataset_service = dataset_service
        self._db_session_factory = db_session_factory

    @step("Prepare dataset", 10)
    def prepare_dataset(self, staged_dataset_id: UUID, task: Task) -> Dataset:
        staged_dataset_path = self._staged_datasets_dir / str(staged_dataset_id) / "dataset"
        if not staged_dataset_path.exists() or not staged_dataset_path.is_dir():
            raise ValueError(f"Staged dataset directory does not exist: {staged_dataset_path}")
        dataset = import_dataset(str(staged_dataset_path))
        if not dataset:
            raise ValueError(f"Failed to import dataset from {staged_dataset_path}")
        target_type = self.__get_sample_by_task(task=task)
        if target_type and target_type != dataset.dtype:
            dataset = dataset.convert_to_schema(target_type)
        return dataset

    @step("Import items from dataset to project")
    def create_items(self, dataset: Dataset, params: ImportDatasetToProjectJobParams) -> None:
        with self._db_session_factory() as session:
            self._dataset_service.set_db_session(session)
            self._label_service.set_db_session(session)
            self._media_service.set_db_session(session)
            labels = self._label_service.list_all(params.project_id)
            label_attr_name = "label" if "label" in dataset.schema.attributes else "labels"
            label_attr = dataset.schema.attributes[label_attr_name]
            converter = DatumaroSampleToGetiAnnotationConverter(
                project_labels=labels,
                label_categories=cast(LabelCategories, label_attr.categories),
                label_mapping=params.labels_mapping,
            )
            logger.info("Found {} labels for project {}", labels, params.project_id)
            size, min_p, max_p = len(dataset), 10, 100
            progress_interval = max(1, size // self.BATCH_PROGRESS_INTERVAL)
            for idx, item in enumerate(dataset):
                if (idx > 0 and idx % progress_interval == 0) or idx == size - 1:
                    self.update_progress(min_p + (idx / size) * (max_p - min_p))
                media = self._media_service.create_image(
                    project_id=params.project_id,
                    name=str(idx).zfill(len(str(size))),
                    format=ImageFormat.JPG,
                    data=item.image.data,
                )
                annotations = converter.convert_sample(item) or None
                user_reviewed = item.user_reviewed if item.user_reviewed is not None else True
                # If there are no annotations (due to filtering), we can consider the item as not reviewed by the user.
                if not annotations:
                    user_reviewed = False
                self._dataset_service.create_dataset_item(
                    project_id=params.project_id,
                    task=params.task,
                    media=media,
                    user_reviewed=user_reviewed,
                    annotations=annotations,
                    subset=DatasetItemSubset(item.subset.name.lower()),
                )

    def execute(self, params: ImportDatasetToProjectJobParams) -> None:
        dataset = self.prepare_dataset(staged_dataset_id=params.staged_dataset_id, task=params.task)
        self.create_items(dataset=dataset, params=params)

    @staticmethod
    def __get_sample_by_task(task: Task) -> type[Sample] | None:
        match task.task_type:
            case TaskType.CLASSIFICATION:
                if not task.exclusive_labels:
                    return MultilabelClassificationSample
                return ClassificationSample
            case TaskType.DETECTION:
                return DetectionSample
            case TaskType.INSTANCE_SEGMENTATION:
                return InstanceSegmentationSample
            case _:
                raise ValueError(f"Unknown task type: {task}")
