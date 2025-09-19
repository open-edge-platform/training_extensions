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
from app.db.schema import DatasetItemDB
from app.repositories import DatasetItemRepository
from app.schemas.dataset_item import DatasetItem, DatasetItemSubset
from app.services.base import InvalidImageError, ResourceNotFoundError, ResourceType
from app.services.mappers.dataset_item_mapper import DatasetItemMapper
from app.utils.images import crop_to_thumbnail

logger = logging.getLogger(__name__)

DEFAULT_THUMBNAIL_SIZE = 256


class DatasetService:
    def __init__(self, data_dir: Path) -> None:
        self.mapper = DatasetItemMapper()
        self.projects_dir = data_dir / "projects"

    def create_dataset_item(self, project_id: UUID, name: str, format: str, size: int, file: BinaryIO) -> DatasetItem:
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
