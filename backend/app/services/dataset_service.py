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
    DatasetItemAnnotations,
    DatasetItemAnnotationsWithSource,
    DatasetItemSubset,
)
from app.schemas.shape import FullImage, Polygon, Rectangle
from app.services.base import AnnotationValidationError, InvalidImageError, ResourceNotFoundError, ResourceType
from app.services.mappers.dataset_item_mapper import DatasetItemMapper
from app.utils.images import crop_to_thumbnail

logger = logging.getLogger(__name__)

DEFAULT_THUMBNAIL_SIZE = 256


class DatasetService:
    def __init__(self) -> None:
        self.mapper = DatasetItemMapper()

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

        dataset_dir = Path(f"data/projects/{str(project_id)}/dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        image.save(dataset_dir / f"{str(dataset_item_id)}.{format}")

        try:
            thumbnail_image = crop_to_thumbnail(
                image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
            )
            if thumbnail_image.mode in ("RGBA", "P"):
                thumbnail_image = thumbnail_image.convert("RGB")
            thumbnail_image.save(dataset_dir / f"{str(dataset_item_id)}-thumb.jpg")
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
                for db in repo.list(limit=limit, offset=offset, start_date=start_date, end_date=end_date)
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
        return Path(f"data/projects/{str(project_id)}/dataset/{dataset_item.id}.{dataset_item.format}")

    def get_dataset_item_thumbnail_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:
        """Get a dataset item thumbnail binary content by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return Path(f"data/projects/{str(project_id)}/dataset/{str(dataset_item.id)}-thumb.jpg")

    def delete_dataset_item(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete a dataset item by its ID"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            deleted = repo.delete(obj_id=str(dataset_item_id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    @staticmethod
    def _validate_annotations_labels(annotations: list[DatasetItemAnnotation], project: ProjectDB) -> None:
        for annotation in annotations:
            for annotation_label in annotation.labels:
                project_label = next((label for label in project.labels if label.id == str(annotation_label.id)), None)
                if project_label is None:
                    raise AnnotationValidationError(f"Label {str(annotation_label.id)} is not found in the project.")

    @staticmethod
    def _validate_annotations(annotations: list[DatasetItemAnnotation], project: ProjectDB) -> None:  # noqa: C901
        if project.task_type == "classification":
            if len(annotations) > 1:
                raise AnnotationValidationError("Classification project doesn't allow more than one annotation.")
            annotation = annotations[0]
            if not isinstance(annotation.shape.root, FullImage):
                raise AnnotationValidationError("Classification project supports only full_image shapes.")
            if project.exclusive_labels and len(annotation.labels) > 1:
                raise AnnotationValidationError(
                    "Multiclass classification project doesn't allow more than one label per annotation."
                )
        if project.task_type == "detection":
            for annotation in annotations:
                if not isinstance(annotation.shape.root, Rectangle):
                    raise AnnotationValidationError("Detection project supports only rectangle shapes.")
                if len(annotation.labels) > 1:
                    raise AnnotationValidationError(
                        "Detection project doesn't allow more than one label per annotation."
                    )
        if project.task_type == "segmentation":
            for annotation in annotations:
                if not isinstance(annotation.shape.root, Polygon):
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
            if isinstance(annotation.shape.root, Rectangle):
                rect = annotation.shape.root
                if rect.x > dataset_item.width or rect.x + rect.width > dataset_item.width:
                    raise AnnotationValidationError("Rectangle coordinates are out of bounds")
                if rect.y > dataset_item.height or rect.y + rect.height > dataset_item.height:
                    raise AnnotationValidationError("Rectangle coordinates are out of bounds")
            if isinstance(annotation.shape.root, Polygon):
                poly = annotation.shape.root
                for point in poly.points:
                    if point.x > dataset_item.width or point.y > dataset_item.height:
                        raise AnnotationValidationError("Polygon points are out of bounds")

    def set_dataset_item_annotations(
        self, project_id: UUID, dataset_item_id: UUID, annotation_data: DatasetItemAnnotations
    ) -> DatasetItemAnnotationsWithSource:
        """Set dataset item annotations"""
        with get_db_session() as db:
            project_repo = ProjectRepository(db=db)
            project = project_repo.get_by_id(str(project_id))
            if project is None:
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            DatasetService._validate_annotations_labels(annotations=annotation_data.annotations, project=project)

            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            DatasetService._validate_annotations_coordinates(
                annotations=annotation_data.annotations, dataset_item=dataset_item
            )

            result = repo.set_annotation_data(
                obj_id=str(dataset_item_id), annotation_data=annotation_data.model_dump(mode="json")
            )
            if not result:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            db.commit()

            return DatasetItemAnnotationsWithSource(
                annotations=DatasetItemAnnotations.model_validate(result.annotation_data).annotations,
                user_reviewed=result.user_reviewed,
                prediction_model_id=result.prediction_model_id,
            )

    def get_dataset_item_annotations(
        self, project_id: UUID, dataset_item_id: UUID
    ) -> DatasetItemAnnotationsWithSource | None:
        """Get the dataset item annotations"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            dataset_item = repo.get_by_id(str(dataset_item_id))
            if not dataset_item:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            if not dataset_item.annotation_data:
                return None
            return DatasetItemAnnotationsWithSource(
                annotations=DatasetItemAnnotations.model_validate(dataset_item.annotation_data).annotations,
                user_reviewed=dataset_item.user_reviewed,
                prediction_model_id=dataset_item.prediction_model_id,
            )

    def delete_dataset_item_annotations(self, project_id: UUID, dataset_item_id: UUID) -> None:
        """Delete the dataset item annotations"""
        with get_db_session() as db:
            repo = DatasetItemRepository(project_id=str(project_id), db=db)
            updated = repo.set_annotation_data(obj_id=str(dataset_item_id), annotation_data=None)
            if not updated:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
            db.commit()
