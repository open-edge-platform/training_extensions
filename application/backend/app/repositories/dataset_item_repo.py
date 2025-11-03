# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from typing import NamedTuple

from sqlalchemy import Select, delete, func, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB
from app.models import DatasetItemSubset


class UpdateDatasetItemAnnotation(NamedTuple):
    annotation_data: list
    user_reviewed: bool
    prediction_model_id: str | None


class DatasetItemRepository:
    """Repository for dataset item-related database operations."""

    def __init__(self, project_id: str, db: Session):
        self.project_id = project_id
        self.db = db

    def _base_select(self) -> Select:
        """Create base select statement filtered by project_id."""
        return select(DatasetItemDB).where(DatasetItemDB.project_id == self.project_id)

    def _apply_date_filters(
        self, stmt: Select, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> Select:
        """Apply date range filters to a select statement."""
        if start_date:
            stmt = stmt.where(DatasetItemDB.created_at >= start_date)
        if end_date:
            stmt = stmt.where(DatasetItemDB.created_at < end_date)
        return stmt

    def save(self, dataset_item_db: DatasetItemDB) -> DatasetItemDB:
        dataset_item_db.updated_at = datetime.now(UTC)
        self.db.add(dataset_item_db)
        self.db.flush()
        return dataset_item_db

    def count(self, start_date: datetime | None = None, end_date: datetime | None = None) -> int:
        stmt = select(func.count()).select_from(DatasetItemDB).where(DatasetItemDB.project_id == self.project_id)
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        return self.db.scalar(stmt) or 0

    def list_items(
        self, limit: int, offset: int, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[DatasetItemDB]:
        stmt = self._base_select()
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = stmt.order_by(DatasetItemDB.created_at.desc()).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def get_earliest(self) -> DatasetItemDB | None:
        """
        Get the earliest dataset item based on creation date.

        This method efficiently uses the multicolumn index on (project_id, created_at)
        for optimal query performance.

        Returns:
            The earliest DatasetItemDB instance or None if no items exist.
        """
        stmt = self._base_select().order_by(DatasetItemDB.created_at.asc()).limit(1)
        return self.db.scalar(stmt)

    def get_by_id(self, obj_id: str) -> DatasetItemDB | None:
        stmt = self._base_select().where(DatasetItemDB.id == obj_id)
        return self.db.scalar(stmt)

    def delete(self, obj_id: str) -> bool:
        stmt = delete(DatasetItemDB).where(
            DatasetItemDB.project_id == self.project_id,
            DatasetItemDB.id == obj_id,
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0  # type: ignore[union-attr]

    def set_annotation_data(self, obj_id: str, annotation_data: list) -> UpdateDatasetItemAnnotation | None:
        stmt = (
            update(DatasetItemDB)
            .returning(
                DatasetItemDB.annotation_data,
                DatasetItemDB.user_reviewed,
                DatasetItemDB.prediction_model_id,
            )
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.id == obj_id,
            )
            .values(
                annotation_data=annotation_data,
                updated_at=datetime.now(UTC),
            )
        )
        result = self.db.execute(stmt)
        row = result.mappings().first()
        return UpdateDatasetItemAnnotation(**row) if row else None

    def delete_annotation_data(self, obj_id: str) -> bool:
        stmt = (
            update(DatasetItemDB)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.id == obj_id,
            )
            .values(
                annotation_data=None,
                updated_at=datetime.now(UTC),
            )
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0  # type: ignore[union-attr]

    def get_subset(self, obj_id: str) -> str | None:
        stmt = (
            select(DatasetItemDB.subset)
            .select_from(DatasetItemDB)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.id == obj_id,
            )
        )
        return self.db.scalar(stmt)

    def set_subset(self, obj_ids: set[str], subset: str) -> int:
        stmt = (
            update(DatasetItemDB)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.id.in_(obj_ids),
            )
            .values(
                subset=subset,
                updated_at=datetime.now(UTC),
            )
        )
        result = self.db.execute(stmt)
        return result.rowcount or 0

    def set_labels(self, dataset_item_id: str, label_ids: set[str]) -> None:
        self.delete_labels(dataset_item_id)

        if label_ids:
            values = [{"dataset_item_id": dataset_item_id, "label_id": label_id} for label_id in label_ids]
            stmt = insert(DatasetItemLabelDB).values(values)
            self.db.execute(stmt)

    def delete_labels(self, dataset_item_id: str) -> None:
        stmt = delete(DatasetItemLabelDB).where(DatasetItemLabelDB.dataset_item_id == dataset_item_id)
        self.db.execute(stmt)

    def list_unassigned_items(self) -> list[DatasetItemLabelDB]:
        stmt = (
            select(DatasetItemLabelDB)
            .join(DatasetItemDB)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.subset == DatasetItemSubset.UNASSIGNED,
            )
        )
        return list(self.db.scalars(stmt).all())

    def get_subset_distribution(self) -> dict[str, int]:
        stmt = (
            select(DatasetItemDB.subset, func.count(DatasetItemDB.id).label("count"))
            .where(DatasetItemDB.project_id == self.project_id)
            .group_by(DatasetItemDB.subset)
        )
        result = self.db.execute(stmt)
        return {row.subset: row.count for row in result}  # type: ignore[misc]
