# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import os.path
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import datumaro.experimental as dm
import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.core.models import TaskType
from app.db.schema import DatasetItemDB
from app.repositories import DatasetItemRepository
from app.schemas.dataset_item import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationsWithSource,
    DatasetItemSubset,
)
from app.schemas.label import LabelBase
from app.schemas.project import ProjectBase
from app.schemas.shape import FullImage, Polygon, Rectangle
from app.services.datumaro_converter import convert_dataset
from app.utils.images import crop_to_thumbnail

from .base import ResourceNotFoundError, ResourceType
from .label_service import LabelService
from .mappers.dataset_item_mapper import DatasetItemMapper
from .project_service import ProjectService

logger = logging.getLogger(__name__)

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


class NotAnnotatedError(Exception):
    """Exception raised when unannotated dataset item annotations are requested."""

    def __init__(self, message: str | None = None):
        msg = message or "Dataset item has not been annotated yet."
        super().__init__(msg)


class SubsetAlreadyAssignedError(Exception):
    """Exception raised when subset is being assigned to a dataset item, which already has one assigned."""

    def __init__(self, message: str | None = None):
        msg = message or "Dataset item has already a subset assigned."
        super().__init__(msg)


class DatasetService:
    def __init__(
        self, data_dir: Path, db_session: Session, project_service: ProjectService, label_service: LabelService
    ) -> None:
        self.mapper = DatasetItemMapper()
        self.projects_dir = data_dir / "projects"
        self._db_session = db_session
        self._project_service = project_service
        self._label_service = label_service

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
        except Exception as e:
            logger.exception("Failed to generate thumbnail image %s", e)

    def create_dataset_item(  # noqa: PLR0913
        self,
        project_id: UUID,
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

        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{dataset_item_id}.{format}"
        image.save(binary_path)

        DatasetService._generate_and_save_thumbnail(image, dataset_dir / f"{dataset_item_id}-thumb.jpg")

        dataset_item = DatasetItemDB(
            id=str(dataset_item_id),
            project_id=str(project_id),
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

        project = self._project_service.get_project_by_id(project_id)

        if annotations is not None:
            DatasetService._validate_annotations_labels(annotations=annotations, labels=project.task.labels)
            DatasetService._validate_annotations(annotations=annotations, project=project)
            DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

            dataset_item.annotation_data = [annotation.model_dump(mode="json") for annotation in annotations]

        repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)
        dataset_item = repo.save(dataset_item)
        return self.mapper.to_schema(dataset_item)

    def count_dataset_items(
        self,
        project_id: UUID,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Get number of available dataset items (within date range if specified)"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)
        return repo.count(start_date=start_date, end_date=end_date)

    def list_dataset_items(
        self,
        project_id: UUID,
        limit: int = 20,
        offset: int = 0,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[DatasetItem]:
        """Get information about available dataset items"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)
        return [
            self.mapper.to_schema(db)
            for db in repo.list_items(limit=limit, offset=offset, start_date=start_date, end_date=end_date)
        ]

    def get_dataset_item_by_id(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItem:
        """Get a dataset item by its ID"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return self.mapper.to_schema(dataset_item)

    def get_dataset_item_binary_path(self, project_id: UUID, dataset_item: DatasetItemDB) -> Path:
        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        return dataset_dir / f"{dataset_item.id}.{dataset_item.format}"

    def get_dataset_item_binary_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item binary content by its ID"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return self.get_dataset_item_binary_path(project_id=project.id, dataset_item=dataset_item)

    def get_dataset_item_thumbnail_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item thumbnail binary content by its ID"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return self.projects_dir / f"{project.id}/dataset/{dataset_item.id}-thumb.jpg"

    def delete_dataset_item(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete a dataset item by its ID"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        try:
            os.remove(dataset_dir / f"{dataset_item.id}.{dataset_item.format}")
        except FileNotFoundError:
            logger.warning(f"Dataset item {dataset_item_id} binary was not found during deletion")
        try:
            os.remove(dataset_dir / f"{dataset_item_id}-thumb.jpg")
        except FileNotFoundError:
            logger.warning(f"Dataset item {dataset_item_id} thumbnail was not found during deletion")

        deleted = repo.delete(obj_id=dataset_item.id)
        if not deleted:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, dataset_item.id)

    @staticmethod
    def _validate_annotations_labels(annotations: list[DatasetItemAnnotation], labels: Sequence[LabelBase]) -> None:
        for annotation in annotations:
            for annotation_label in annotation.labels:
                project_label = next((label for label in labels if label.id == annotation_label.id), None)
                if project_label is None:
                    raise AnnotationValidationError(f"Label {str(annotation_label.id)} is not found in the project.")

    @staticmethod
    def _validate_annotations(annotations: list[DatasetItemAnnotation], project: ProjectBase) -> None:  # noqa: C901
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
        annotations: list[DatasetItemAnnotation], dataset_item: DatasetItemDB
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
        self, project_id: UUID, dataset_item_id: UUID, annotations: list[DatasetItemAnnotation]
    ) -> DatasetItemAnnotationsWithSource:
        """Set dataset item annotations"""
        project = self._project_service.get_project_by_id(project_id)

        DatasetService._validate_annotations_labels(annotations=annotations, labels=project.task.labels)
        DatasetService._validate_annotations(annotations=annotations, project=project)

        repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

        result = repo.set_annotation_data(
            obj_id=str(dataset_item_id),
            annotation_data=[annotation.model_dump(mode="json") for annotation in annotations],
        )
        if not result:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

        return DatasetItemAnnotationsWithSource(
            annotations=[DatasetItemAnnotation.model_validate(annotation) for annotation in result.annotation_data],
            user_reviewed=result.user_reviewed,
            prediction_model_id=result.prediction_model_id,
        )

    def get_dataset_item_annotations(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItemAnnotationsWithSource:
        """Get the dataset item annotations"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        dataset_item = repo.get_by_id(str(dataset_item_id))
        if not dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        if not dataset_item.annotation_data:
            raise NotAnnotatedError
        return DatasetItemAnnotationsWithSource(
            annotations=[
                DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
            ],
            user_reviewed=dataset_item.user_reviewed,
            prediction_model_id=dataset_item.prediction_model_id,
        )

    def delete_dataset_item_annotations(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete the dataset item annotations"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        updated = repo.delete_annotation_data(obj_id=str(dataset_item_id))
        if not updated:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def assign_dataset_item_subset(
        self, project_id: UUID, dataset_item_id: UUID, subset: DatasetItemSubset
    ) -> DatasetItem:
        """Assign dataset item subset"""
        project = self._project_service.get_project_by_id(project_id)
        repo = DatasetItemRepository(project_id=str(project.id), db=self._db_session)
        db_subset = repo.get_subset(str(dataset_item_id))
        if db_subset is None:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        if db_subset != DatasetItemSubset.UNASSIGNED:
            raise SubsetAlreadyAssignedError
        repo.set_subset(obj_id=str(dataset_item_id), subset=subset)
        return self.get_dataset_item_by_id(project_id=project_id, dataset_item_id=dataset_item_id)

    def get_dm_dataset(self, project_id: UUID) -> dm.Dataset:
        repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)

        def _get_dataset_items(offset: int, limit: int) -> list[DatasetItemDB]:
            return repo.list_items(limit=limit, offset=offset)

        def _get_image_path(item: DatasetItemDB) -> str:
            return str(self.get_dataset_item_binary_path(project_id=project_id, dataset_item=item))

        project = self._project_service.get_project_by_id(project_id=project_id)
        labels = self._label_service.list_all(project_id=project_id)
        return convert_dataset(
            project=project, labels=labels, get_dataset_items=_get_dataset_items, get_image_path=_get_image_path
        )
