# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from app.db import get_db_session
from app.db.schema import DatasetItemDB, ProjectDB
from app.repositories import DatasetItemRepository, ProjectRepository
from app.schemas.dataset_item import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationsWithSource,
    DatasetItemSubset,
)
from app.schemas.project import TaskType
from app.schemas.shape import FullImage, Polygon, Rectangle
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.mappers.dataset_item_mapper import DatasetItemMapper
from app.utils.images import crop_to_thumbnail

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


class DatasetService:
    def __init__(self, data_dir: Path) -> None:
        self.mapper = DatasetItemMapper()
        self.projects_dir = data_dir / "projects"

    def create_dataset_item(
        self, project_id: UUID, name: str, format: str, size: int, file: BinaryIO, user_reviewed: bool
    ) -> DatasetItem:
        """Creates a new dataset item"""
        file.seek(0)
        try:
            image: Image.Image = Image.open(file)
        except UnidentifiedImageError:
            raise InvalidImageError
        dataset_item_id = uuid4()
        dataset_item = DatasetItemDB(
            id=str(dataset_item_id),
            project_id=str(project_id),
            name=name,
            format=format,
            width=image.width,
            height=image.height,
            size=size,
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=user_reviewed,
        )

        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        image.save(dataset_dir / f"{dataset_item_id}.{format}")

        try:
            thumbnail_image = crop_to_thumbnail(
                image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
            )
            if thumbnail_image.mode in ("RGBA", "P"):
                thumbnail_image = thumbnail_image.convert("RGB")
            thumbnail_image.save(dataset_dir / f"{dataset_item_id}-thumb.jpg")
        except Exception as e:
            logger.exception("Failed to generate thumbnail image %s", e)

        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            result = self.mapper.to_schema(repo.save(dataset_item))
            db.commit()
        return result

    def count_dataset_items(
        self,
        project_id: UUID,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Get number of available dataset items (within date range if specified)"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
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
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            return [
                self.mapper.to_schema(db)
                for db in repo.list_items(limit=limit, offset=offset, start_date=start_date, end_date=end_date)
            ]

    def get_dataset_item_by_id(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItem:
        """Get a dataset item by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            return self.mapper.to_schema(dataset_item)

    def get_dataset_item_binary_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item binary content by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return self.projects_dir / f"{project_id}/dataset/{dataset_item.id}.{dataset_item.format}"

    def get_dataset_item_thumbnail_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item thumbnail binary content by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return self.projects_dir / f"{project_id}/dataset/{dataset_item.id}-thumb.jpg"

    def delete_dataset_item(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete a dataset item by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

            dataset_dir = self.projects_dir / f"{project_id}/dataset"
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
            db.commit()

    @staticmethod
    def _validate_annotations_labels(annotations: list[DatasetItemAnnotation], project: ProjectDB) -> None:
        for annotation in annotations:
            for annotation_label in annotation.labels:
                project_label = next((label for label in project.labels if label.id == str(annotation_label.id)), None)
                if project_label is None:
                    raise AnnotationValidationError(f"Label {str(annotation_label.id)} is not found in the project.")

    @staticmethod
    def _validate_annotations(annotations: list[DatasetItemAnnotation], project: ProjectDB) -> None:  # noqa: C901
        match project.task_type:
            case TaskType.CLASSIFICATION:
                if len(annotations) > 1:
                    raise AnnotationValidationError("Classification project doesn't allow more than one annotation.")
                annotation = annotations[0]
                if not isinstance(annotation.shape, FullImage):
                    raise AnnotationValidationError("Classification project supports only full_image shapes.")
                if project.exclusive_labels and len(annotation.labels) > 1:
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
            case TaskType.SEGMENTATION:
                for annotation in annotations:
                    if not isinstance(annotation.shape, Polygon):
                        raise AnnotationValidationError("Segmentation project supports only polygon shapes.")
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
        with get_db_session() as db:
            project_repo = ProjectRepository(db=db)
            project = project_repo.get_by_id(str(project_id))
            if project is None:
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            DatasetService._validate_annotations_labels(annotations=annotations, project=project)
            DatasetService._validate_annotations(annotations=annotations, project=project)

            repo = DatasetItemRepository(project_id=str(project_id), db=db)
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
            db.commit()

            return DatasetItemAnnotationsWithSource(
                annotations=[DatasetItemAnnotation.model_validate(annotation) for annotation in result.annotation_data],
                user_reviewed=result.user_reviewed,
                prediction_model_id=result.prediction_model_id,
            )

    def get_dataset_item_annotations(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItemAnnotationsWithSource:
        """Get the dataset item annotations"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            return DatasetItemAnnotationsWithSource(
                annotations=[
                    DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
                ],
                user_reviewed=dataset_item.user_reviewed,
                prediction_model_id=dataset_item.prediction_model_id,
            )

    def delete_dataset_item_annotations(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete the dataset item annotations"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            updated = repo.set_annotation_data(obj_id=str(dataset_item_id), annotation_data=[])
            if not updated:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            db.commit()
