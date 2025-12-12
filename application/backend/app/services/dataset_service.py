# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import os.path
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import datumaro.experimental as dm
import numpy as np
from datumaro.experimental.export_import import export_dataset
from loguru import logger
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetRevisionDB
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    FullImage,
    Label,
    Polygon,
    Project,
    Rectangle,
    Task,
    TaskType,
)
from app.models.dataset_revision import DatasetRevision
from app.repositories import DatasetItemRepository, DatasetRevisionRepository
from app.services.datumaro_converter import convert_dataset
from app.utils.images import crop_to_thumbnail

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType
from .label_service import LabelService

DEFAULT_THUMBNAIL_SIZE = 256


class AnnotationValidationError(Exception):
    """Exception raised when dataset annotation validation has failed."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidImageError(Exception):
    """Exception raised when invalid image is used to create a dataset item."""

    def __init__(self, message: str | None = None):
        msg = message or "Invalid image has been passed while creating a dataset item."
        super().__init__(msg)


class SubsetAlreadyAssignedError(Exception):
    """Exception raised when subset is being assigned to a dataset item, which already has one assigned."""

    def __init__(self, message: str | None = None):
        msg = message or "Dataset item has already a subset assigned."
        super().__init__(msg)


@dataclass(frozen=True)
class DatasetItemFilters:
    limit: int = 20
    offset: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    annotation_status: DatasetItemAnnotationStatus | None = None
    label_ids: list[UUID] | None = None
    subset: str | None = None


class DatasetService(BaseSessionManagedService):
    def __init__(
        self,
        data_dir: Path,
        label_service: LabelService,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)

        self.projects_dir = data_dir / "projects"
        self._label_service = label_service
        self.register_managed_services(label_service)

    @staticmethod
    def _read_image_from_ndarray(data: np.ndarray) -> Image.Image:
        return Image.fromarray(data)

    @staticmethod
    def _read_image_from_binary(data: BinaryIO | BytesIO) -> Image.Image:
        data.seek(0)
        try:
            return Image.open(data)
        except UnidentifiedImageError:
            raise InvalidImageError

    @staticmethod
    def _generate_and_save_thumbnail(image: Image.Image, path: Path) -> None:
        try:
            thumbnail_image = crop_to_thumbnail(
                image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
            )
            if thumbnail_image.mode in ("RGBA", "P"):
                thumbnail_image = thumbnail_image.convert("RGB")
            thumbnail_image.save(path)
        except Exception:
            logger.exception("Failed to generate thumbnail image")

    def create_dataset_item(  # noqa: PLR0913
        self,
        project: Project,
        name: str,
        format: str,
        data: Image.Image | np.ndarray | BinaryIO | BytesIO,
        user_reviewed: bool,
        source_id: UUID | None = None,
        prediction_model_id: UUID | None = None,
        annotations: list[DatasetItemAnnotation] | None = None,
    ) -> DatasetItem:
        """Creates a new dataset item"""
        dataset_item_id = uuid4()
        match data:
            case Image.Image():
                image = data
            case np.ndarray():
                image = self._read_image_from_ndarray(data)
            case _:
                image = self._read_image_from_binary(data)

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{dataset_item_id}.{format}"
        image.save(binary_path)

        DatasetService._generate_and_save_thumbnail(image, dataset_dir / f"{dataset_item_id}-thumb.jpg")

        dataset_item = DatasetItemDB(
            id=str(dataset_item_id),
            project_id=str(project.id),
            name=name,
            format=format,
            width=image.width,
            height=image.height,
            size=os.path.getsize(binary_path),
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=user_reviewed,
            source_id=str(source_id) if source_id is not None else None,
            prediction_model_id=str(prediction_model_id) if prediction_model_id is not None else None,
        )

        if annotations is not None:
            labels = self._label_service.list_all(project_id=project.id)
            DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)
            DatasetService._validate_annotations(annotations=annotations, project=project)
            DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

            dataset_item.annotation_data = [annotation.model_dump(mode="json") for annotation in annotations]

        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        db_dataset_item = repo.save(dataset_item)
        if annotations is not None:
            repo.set_labels(
                dataset_item_id=str(dataset_item_id),
                label_ids={str(label.id) for annotation in annotations for label in annotation.labels},
            )
        return DatasetItem.model_validate(db_dataset_item)

    def count_dataset_items(
        self,
        project: Project,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[UUID] | None = None,
        subset: str | None = None,
    ) -> int:
        """Get number of available dataset items (within date range if specified)"""
        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in label_ids] if label_ids else None
        return repo.count(
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=label_ids_str,
            subset=subset,
        )

    def list_dataset_items(
        self,
        project_id: UUID,
        filters: DatasetItemFilters | None = None,
    ) -> list[DatasetItem]:
        """Get information about available dataset items"""
        if filters is None:
            filters = DatasetItemFilters()
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return [
            DatasetItem.model_validate(db)
            for db in repo.list_items(
                limit=filters.limit,
                offset=filters.offset,
                start_date=filters.start_date,
                end_date=filters.end_date,
                annotation_status=filters.annotation_status,
                label_ids=label_ids_str,
                subset=filters.subset,
            )
        ]

    def get_dataset_item_by_id(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItem:
        """Get a dataset item by its ID"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        db_dataset_item = repo.get_by_id(str(dataset_item_id))
        if not db_dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return DatasetItem.model_validate(db_dataset_item)

    def get_dataset_item_binary_path(self, project_id: UUID, dataset_item: DatasetItemDB | DatasetItem) -> Path:
        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        return dataset_dir / f"{dataset_item.id}.{dataset_item.format}"

    def get_dataset_item_binary_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item binary content by its ID"""
        dataset_item = self.get_dataset_item_by_id(project_id=project_id, dataset_item_id=dataset_item_id)
        return self.get_dataset_item_binary_path(project_id=project_id, dataset_item=dataset_item)

    def get_dataset_item_thumbnail_path_by_id(self, project: Project, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item thumbnail binary content by its ID"""
        dataset_item = self.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
        return self.projects_dir / f"{project.id}/dataset/{dataset_item.id}-thumb.jpg"

    def delete_dataset_item(self, project: Project, dataset_item_id: UUID) -> None:
        """Delete a dataset item by its ID"""
        dataset_item = self.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        try:
            os.remove(dataset_dir / f"{dataset_item.id}.{dataset_item.format}")
        except FileNotFoundError:
            logger.warning("Dataset item {} binary was not found during deletion", dataset_item_id)
        try:
            os.remove(dataset_dir / f"{dataset_item_id}-thumb.jpg")
        except FileNotFoundError:
            logger.warning("Dataset item {} thumbnail was not found during deletion", dataset_item_id)

        repo.delete(obj_id=str(dataset_item.id))

    @staticmethod
    def _validate_annotations_labels(annotations: list[DatasetItemAnnotation], labels: Sequence[Label]) -> None:
        for annotation in annotations:
            for annotation_label in annotation.labels:
                project_label = next((label for label in labels if label.id == annotation_label.id), None)
                if project_label is None:
                    raise AnnotationValidationError(f"Label {str(annotation_label.id)} is not found in the project.")

    @staticmethod
    def _validate_annotations(annotations: list[DatasetItemAnnotation], project: Project) -> None:  # noqa: C901
        match project.task.task_type:
            case TaskType.CLASSIFICATION:
                if len(annotations) > 1:
                    raise AnnotationValidationError("Classification project doesn't allow more than one annotation.")
                annotation = annotations[0]
                if not isinstance(annotation.shape, FullImage):
                    raise AnnotationValidationError("Classification project supports only full_image shapes.")
                if project.task.exclusive_labels and len(annotation.labels) > 1:
                    raise AnnotationValidationError(
                        "Multiclass classification project doesn't allow more than one label per annotation."
                    )
            case TaskType.DETECTION:
                for annotation in annotations:
                    if not isinstance(annotation.shape, Rectangle):
                        raise AnnotationValidationError("Detection project supports only rectangle shapes.")
                    if len(annotation.labels) > 1:
                        raise AnnotationValidationError(
                            "Detection project doesn't allow more than one label per annotation."
                        )
            case TaskType.INSTANCE_SEGMENTATION:
                for annotation in annotations:
                    if not isinstance(annotation.shape, Polygon):
                        raise AnnotationValidationError("Instance Segmentation project supports only polygon shapes.")
                    if len(annotation.labels) > 1:
                        raise AnnotationValidationError(
                            "Segmentation project doesn't allow more than one label per annotation."
                        )

    @staticmethod
    def _validate_annotations_coordinates(
        annotations: list[DatasetItemAnnotation], dataset_item: DatasetItem | DatasetItemDB
    ) -> None:
        for annotation in annotations:
            if isinstance(annotation.shape, Rectangle):
                rect = annotation.shape
                if rect.x > dataset_item.width or rect.x + rect.width > dataset_item.width:
                    raise AnnotationValidationError("Rectangle coordinates are out of bounds")
                if rect.y > dataset_item.height or rect.y + rect.height > dataset_item.height:
                    raise AnnotationValidationError("Rectangle coordinates are out of bounds")
            if isinstance(annotation.shape, Polygon):
                poly = annotation.shape
                for point in poly.points:
                    if point.x > dataset_item.width or point.y > dataset_item.height:
                        raise AnnotationValidationError("Polygon points are out of bounds")

    def set_dataset_item_annotations(
        self, project: Project, dataset_item_id: UUID, annotations: list[DatasetItemAnnotation], user_reviewed: bool
    ) -> DatasetItem:
        """
        Set dataset item annotations

        Args:
            project: The project to which the dataset item belongs.
            dataset_item_id: The ID of the dataset item.
            annotations: The list of annotations to set. Overwrites existing annotations, if any.
            user_reviewed: Whether the annotations have been reviewed by a user.

        Returns:
            The updated dataset item.
        """
        labels = self._label_service.list_all(project_id=project.id)
        DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)
        DatasetService._validate_annotations(annotations=annotations, project=project)

        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        dataset_item = self.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)

        DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

        result = repo.set_annotation_data(
            obj_id=str(dataset_item_id),
            annotation_data=[annotation.model_dump(mode="json") for annotation in annotations],
            user_reviewed=user_reviewed,
        )
        if not result:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

        repo.set_labels(
            dataset_item_id=str(dataset_item_id),
            label_ids={str(label.id) for annotation in annotations for label in annotation.labels},
        )
        return self.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)

    def delete_dataset_item_annotations(self, project: Project, dataset_item_id: UUID) -> None:
        """Delete the dataset item annotations"""
        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        updated = repo.delete_annotation_data(obj_id=str(dataset_item_id))
        if not updated:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        repo.delete_labels(dataset_item_id=str(dataset_item_id))

    def assign_dataset_item_subset(
        self, project_id: UUID, dataset_item_id: UUID, subset: DatasetItemSubset
    ) -> DatasetItem:
        """Assign dataset item subset"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        db_subset = repo.get_subset(str(dataset_item_id))
        if db_subset is None:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        if db_subset != DatasetItemSubset.UNASSIGNED:
            raise SubsetAlreadyAssignedError
        repo.set_subset(obj_ids={str(dataset_item_id)}, subset=subset)
        return self.get_dataset_item_by_id(project_id=project_id, dataset_item_id=dataset_item_id)

    def get_dm_dataset(
        self, project_id: UUID, task: Task, annotation_status: DatasetItemAnnotationStatus | None
    ) -> dm.Dataset:
        def _get_dataset_items(offset: int, limit: int) -> list[DatasetItem]:
            return self.list_dataset_items(
                project_id=project_id,
                filters=DatasetItemFilters(limit=limit, offset=offset, annotation_status=annotation_status),
            )

        def _get_image_path(item: DatasetItem) -> str:
            return str(self.get_dataset_item_binary_path(project_id=project_id, dataset_item=item))

        labels = self._label_service.list_all(project_id=project_id)
        return convert_dataset(
            task=task,
            labels=labels,
            get_dataset_items=_get_dataset_items,
            get_image_path=_get_image_path,
        )

    def save_revision(self, project_id: UUID, dataset: dm.Dataset) -> UUID:
        """
        Saves the dataset as a new revision.

        Creates a new dataset revision entry in the database and exports the dataset
        to a zip file in the project's revisions directory.

        Args:
            project_id: The UUID of the project to save the revision for.
            dataset: The Datumaro dataset to export.

        Returns:
            UUID: The UUID of the newly created dataset revision.
        """
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision_db = revision_repo.save(
            DatasetRevisionDB(
                project_id=str(project_id),
            )
        )
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / revision_db.id
        logger.info("Saving dataset revision '{}' to '{}'", revision_db.id, revision_path)
        export_dataset(
            dataset=dataset,
            output_path=revision_path,
            export_images=True,
            as_zip=True,
        )
        return UUID(revision_db.id)

    def get_dataset_revision(self, project_id: UUID, revision_id: UUID) -> DatasetRevision:
        """
        Get a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            DatasetRevision: The dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision = revision_repo.get_by_id(str(revision_id))
        if revision is None or revision.project_id != str(project_id):
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))
        return self._to_dataset_revision(dataset_db=revision)

    def delete_dataset_revision_files(self, project_id: UUID, revision_id: UUID) -> None:
        """
        Delete the files associated with a dataset revision.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision = self.get_dataset_revision(project_id, revision_id)
        if revision.files_deleted:
            logger.info("Files for dataset revision '{}' already deleted", revision_id)
            return

        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id)
        if revision_path.exists():
            shutil.rmtree(revision_path)
            logger.info("Deleted dataset revision files at '{}'", revision_path)

        # Mark as deleted in the database
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision_db = revision_repo.get_by_id(str(revision_id))
        if revision_db:
            revision_db.files_deleted = True
            revision_repo.save(revision_db)

    @staticmethod
    def _to_dataset_revision(dataset_db: DatasetRevisionDB) -> DatasetRevision:
        """Convert database model to DatasetRevision."""
        return DatasetRevision.model_validate(dataset_db, from_attributes=True)
