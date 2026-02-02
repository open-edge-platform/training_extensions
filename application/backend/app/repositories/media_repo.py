# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import CursorResult, Select, delete, func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB
from app.models import DatasetItemAnnotationStatus


class MediaRepository:
    """Repository for media-related database operations."""

    def __init__(self, project_id: str, db: Session):
        self.project_id = project_id
        self.db = db

    def _base_select(self) -> Select:
        """Create base select statement filtered by project_id."""
        return select(MediaDB).where(MediaDB.project_id == self.project_id)

    @staticmethod
    def _apply_date_filters(
        stmt: Select, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> Select:
        """Apply date range filters to a select statement."""
        if start_date:
            stmt = stmt.where(MediaDB.created_at >= start_date)
        if end_date:
            stmt = stmt.where(MediaDB.created_at < end_date)
        return stmt

    @staticmethod
    def _apply_annotation_status_filter(stmt: Select, annotation_status: str | None = None) -> Select:
        """Apply annotation status filter to a select statement."""
        if annotation_status == DatasetItemAnnotationStatus.UNANNOTATED:
            stmt = stmt.where(DatasetItemDB.annotation_data.is_(None))
        elif annotation_status == DatasetItemAnnotationStatus.REVIEWED:
            stmt = stmt.where(DatasetItemDB.user_reviewed.is_(True))
        elif annotation_status == DatasetItemAnnotationStatus.TO_REVIEW:
            stmt = stmt.where(DatasetItemDB.user_reviewed.is_(False))
        return stmt

    @staticmethod
    def _apply_subset_filter(stmt: Select, subset: str | None = None) -> Select:
        """Apply subset filter to a select statement."""
        if subset is not None:
            stmt = stmt.where(DatasetItemDB.subset == subset)
        return stmt

    def save(self, media_db: MediaDB) -> MediaDB:
        media_db.updated_at = datetime.now(UTC)
        self.db.add(media_db)
        self.db.flush()
        return media_db

    def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subset: str | None = None,
    ) -> int:
        stmt = (
            select(func.count(func.distinct(MediaDB.id)))
            .select_from(MediaDB)
            .join(DatasetItemDB)
            .where(DatasetItemDB.id == MediaDB.id, DatasetItemDB.project_id == MediaDB.project_id)
            .where(MediaDB.project_id == self.project_id)
        )
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = self._apply_annotation_status_filter(stmt, annotation_status)
        stmt = self._apply_subset_filter(stmt, subset)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids))
        return self.db.scalar(stmt) or 0

    def list_items(
        self,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subset: str | None = None,
    ) -> list[MediaDB]:
        stmt = self._base_select().join(DatasetItemDB).where(DatasetItemDB.id == MediaDB.id)
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = self._apply_annotation_status_filter(stmt, annotation_status)
        stmt = self._apply_subset_filter(stmt, subset)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids)).distinct()
        stmt = stmt.order_by(MediaDB.created_at.desc()).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def get_earliest(self) -> MediaDB | None:
        """
        Get the earliest media based on creation date.

        This method efficiently uses the multicolumn index on (project_id, created_at)
        for optimal query performance.

        Returns:
            The earliest MediaDB instance or None if no items exist.
        """
        stmt = self._base_select().order_by(MediaDB.created_at.asc()).limit(1)
        return self.db.scalar(stmt)

    def get_by_id(self, obj_id: str) -> MediaDB | None:
        stmt = self._base_select().where(MediaDB.id == obj_id)
        return self.db.scalar(stmt)

    def delete(self, obj_id: str) -> bool:
        stmt = delete(MediaDB).where(
            MediaDB.project_id == self.project_id,
            MediaDB.id == obj_id,
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0
