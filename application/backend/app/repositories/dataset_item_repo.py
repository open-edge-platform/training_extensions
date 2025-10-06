# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from typing import NamedTuple, Literal

from sqlalchemy import Select, delete, func, select, update
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB
from app.schemas.dataset_item import AnnotationStatus


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

    def _apply_annotation_status_filter(
        self, stmt: Select, annotation_status: AnnotationStatus | None
    ) -> Select:
        """Apply annotation status filter to SQL query statement.

        Args:
            stmt: Select statement to apply filters to.
            annotation_status: Filter criteria for annotation status. Valid values are:
                - "unannotated": Items with no annotation data
                - "reviewed": Items with annotation data that have been user reviewed
                - "to_review": Items with annotation data pending user review
                - None: No filtering applied

        Returns:
            Select: Modified SQLAlchemy Select statement with annotation status filters applied.
        """
        if annotation_status == AnnotationStatus.UNANNOTATED:
            stmt = stmt.where(DatasetItemDB.annotation_data.is_(None))
        elif annotation_status == AnnotationStatus.REVIEWED:
            stmt = stmt.where(
                DatasetItemDB.annotation_data.is_not(None),
                DatasetItemDB.user_reviewed.is_(True),
            )
        elif annotation_status == AnnotationStatus.TO_REVIEW:
            stmt = stmt.where(
                DatasetItemDB.annotation_data.is_not(None),
                DatasetItemDB.user_reviewed.is_(False),
            )
        return stmt

    def save(self, dataset_item_db: DatasetItemDB) -> DatasetItemDB:
        """Save dataset item to database with updated timestamp.

        Args:
            dataset_item_db: DatasetItemDB instance to be saved.

        Returns:
            DatasetItemDB: The saved DatasetItemDB instance.
        """
        dataset_item_db.updated_at = datetime.now(UTC)
        self.db.add(dataset_item_db)
        self.db.flush()
        return dataset_item_db

    def count(
        self, start_date: datetime | None = None, end_date: datetime | None = None, annotation_status: AnnotationStatus | None = None,
    ) -> int:
        """Count dataset items matching specified filters.

        Args:
            start_date: Optional start date for filtering items by creation date.
            end_date: Optional end date for filtering items by creation date.
            annotation_status: Optional annotation status filter.

        Returns:
            int: Count of dataset items matching the specified filters.
        """
        stmt = select(func.count()).select_from(DatasetItemDB).where(DatasetItemDB.project_id == self.project_id)
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = self._apply_annotation_status_filter(stmt, annotation_status)
        return self.db.scalar(stmt) or 0

    def list_items(
        self, limit: int, offset: int, start_date: datetime | None = None, end_date: datetime | None = None, annotation_status: AnnotationStatus | None = None,
    ) -> list[DatasetItemDB]:
        """Retrieve paginated list of dataset items with optional filtering.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip for pagination.
            start_date: Optional start date for creation date filtering.
            end_date: Optional end date for creation date filtering.
            annotation_status: Optional annotation status filter.

        Returns:
            list[DatasetItemDB]: List of DatasetItemDB instances ordered by creation date (descending).
        """
        stmt = self._base_select()
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = self._apply_annotation_status_filter(stmt, annotation_status)
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
