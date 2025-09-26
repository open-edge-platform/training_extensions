# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import NamedTuple

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB


class UpdateDatasetItemAnnotation(NamedTuple):
    annotation_data: list
    user_reviewed: bool
    prediction_model_id: str | None


class DatasetItemRepository:
    """Repository for dataset item-related database operations."""

    def __init__(self, project_id: str, db: Session):
        self.project_id = project_id
        self.db = db

    def save(self, dataset_item_db: DatasetItemDB) -> DatasetItemDB:
        dataset_item_db.updated_at = datetime.now()
        self.db.add(dataset_item_db)
        self.db.flush()
        return dataset_item_db

    def count(self, start_date: datetime | None = None, end_date: datetime | None = None) -> int:
        query = self.db.query(DatasetItemDB).filter(DatasetItemDB.project_id == self.project_id)
        if start_date:
            query = query.filter(DatasetItemDB.created_at >= start_date)
        if end_date:
            query = query.filter(DatasetItemDB.created_at < end_date)
        return query.count()

    def list_items(
        self, limit: int, offset: int, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[DatasetItemDB]:
        query = self.db.query(DatasetItemDB).filter(DatasetItemDB.project_id == self.project_id)
        if start_date:
            query = query.filter(DatasetItemDB.created_at >= start_date)
        if end_date:
            query = query.filter(DatasetItemDB.created_at < end_date)
        return query.slice(offset, offset + limit).all()

    def get_earliest(self) -> DatasetItemDB | None:
        """
        Get the earliest dataset item based on creation date.

        This method efficiently uses the multicolumn index on (project_id, created_at)
        for optimal query performance.

        Returns:
            The earliest DatasetItemDB instance or None if no items exist.
        """
        return (
            self.db.query(DatasetItemDB)
            .filter(DatasetItemDB.project_id == self.project_id)
            .order_by(DatasetItemDB.created_at.asc())
            .first()
        )

    def get_by_id(self, obj_id: str) -> DatasetItemDB | None:
        return (
            self.db.query(DatasetItemDB)
            .filter(DatasetItemDB.project_id == self.project_id)
            .filter(DatasetItemDB.id == obj_id)
            .one_or_none()
        )

    def delete(self, obj_id: str) -> bool:
        return (
            self.db.query(DatasetItemDB)
            .filter(DatasetItemDB.project_id == self.project_id)
            .filter(DatasetItemDB.id == obj_id)
            .delete()
            > 0
        )

    def set_annotation_data(self, obj_id: str, annotation_data: list) -> UpdateDatasetItemAnnotation | None:
        result = self.db.execute(
            update(DatasetItemDB)
            .returning(DatasetItemDB.annotation_data, DatasetItemDB.user_reviewed, DatasetItemDB.prediction_model_id)
            .filter(DatasetItemDB.project_id == self.project_id)
            .filter(DatasetItemDB.id == obj_id)
            .values({DatasetItemDB.annotation_data: annotation_data, DatasetItemDB.updated_at: datetime.now()})
        )
        return next((UpdateDatasetItemAnnotation(**row) for row in result.mappings().all()), None)
