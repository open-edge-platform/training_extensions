# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, Select, delete, func, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB
from app.models import DatasetItemSubset

from .filters import _apply_annotation_status_filter, _apply_subset_filter


class DatasetItemRepository:
    """Repository for dataset item-related database operations."""

    def __init__(self, project_id: str, db: Session):
        self.project_id = project_id
        self.db = db

    def _base_select(self) -> Select:
        """Create base select statement filtered by project_id."""
        return select(DatasetItemDB).where(DatasetItemDB.project_id == self.project_id)

    @staticmethod
    def _apply_date_filters(
        stmt: Select, start_date: datetime | None = None, end_date: datetime | None = None
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

    def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subset: str | None = None,
    ) -> int:
        # When the query involves a JOIN (e.g. when filtering by labels), count distinct items to avoid duplicates
        if label_ids:
            select_fn = func.count(func.distinct(DatasetItemDB.id))
        else:
            select_fn = func.count()
        stmt = select(select_fn).select_from(DatasetItemDB).where(DatasetItemDB.project_id == self.project_id)
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subset)
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
    ) -> list[DatasetItemDB]:
        stmt = self._base_select()
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subset)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids)).distinct()
        stmt = stmt.order_by(DatasetItemDB.created_at.desc()).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def list_items_with_media(
        self,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subset: str | None = None,
    ) -> list[tuple[DatasetItemDB, MediaDB]]:
        stmt = (
            select(DatasetItemDB, MediaDB)
            .where(DatasetItemDB.project_id == self.project_id)
            .join(MediaDB)
            .where(MediaDB.id == DatasetItemDB.id, MediaDB.project_id == DatasetItemDB.project_id)
        )
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subset)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids)).distinct()
        stmt = stmt.order_by(DatasetItemDB.created_at.desc()).offset(offset).limit(limit)
        return [(dataset_item, media) for (dataset_item, media) in self.db.execute(stmt).all()]

    def get_by_id(self, obj_id: str) -> DatasetItemDB | None:
        stmt = self._base_select().where(DatasetItemDB.id == obj_id)
        return self.db.scalar(stmt)

    def delete(self, obj_id: str) -> bool:
        stmt = delete(DatasetItemDB).where(
            DatasetItemDB.project_id == self.project_id,
            DatasetItemDB.id == obj_id,
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

    def set_annotation_data(
        self, obj_id: str, annotation_data: list, user_reviewed: bool, prediction_model_id: str | None
    ) -> bool:
        stmt = (
            update(DatasetItemDB)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.id == obj_id,
            )
            .values(
                annotation_data=annotation_data,
                user_reviewed=user_reviewed,
                prediction_model_id=prediction_model_id,
                updated_at=datetime.now(UTC),
            )
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

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
                user_reviewed=False,
                prediction_model_id=None,
            )
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

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
        result = cast(CursorResult, self.db.execute(stmt))
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

    def find_items_by_label_id(self, label_id: str) -> list[DatasetItemDB]:
        """Find all dataset items that reference a given label via the dataset_items_labels table."""
        stmt = (
            self._base_select()
            .join(DatasetItemLabelDB, DatasetItemLabelDB.dataset_item_id == DatasetItemDB.id)
            .where(DatasetItemLabelDB.label_id == label_id)
        )
        return list(self.db.scalars(stmt).all())

    def delete_label_from_items(self, label_id: str) -> None:
        """Delete a label reference from the dataset_items_labels table for all items."""
        stmt = delete(DatasetItemLabelDB).where(
            DatasetItemLabelDB.label_id == label_id,
        )
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

    def has_all_subsets_assigned(self) -> bool:
        """Return True if there is at least one dataset item for each of TRAINING, VALIDATION, and TESTING subsets."""
        stmt = select(func.distinct(DatasetItemDB.subset)).where(
            DatasetItemDB.project_id == self.project_id,
        )
        present_subsets = set(self.db.scalars(stmt).all())
        required_subsets = {DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING}
        return required_subsets.issubset(present_subsets)

    def get_subset_distribution(self) -> dict[str, int]:
        stmt = (
            select(DatasetItemDB.subset, func.count(DatasetItemDB.id).label("count"))
            .where(DatasetItemDB.project_id == self.project_id)
            .group_by(DatasetItemDB.subset)
        )
        result = self.db.execute(stmt)
        return {row.subset: row.count for row in result}  # type: ignore[misc]

    def get_statistics(self) -> dict[str, Any]:
        # Media Counts (images, videos, video frames):
        media_counts_stmt = (
            select(
                MediaDB.type, func.count(MediaDB.id).label("count"), func.sum(MediaDB.frame_count).label("frame_count")
            )
            .where(MediaDB.project_id == self.project_id)
            .group_by(MediaDB.type)
        )
        result = self.db.execute(media_counts_stmt)
        rows = list(result)
        statistics: dict[str, Any] = {f"{row.type}s": cast(int, row.count) for row in rows}
        statistics["video_frames"] = int(sum(row.frame_count or 0 for row in rows))

        # Annotation Counts (annotated images and video frames):
        annotated_images_frames_stmt = (
            select(MediaDB.type, func.count(DatasetItemDB.id).label("count"))
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.annotation_data.isnot(None),
                DatasetItemDB.user_reviewed,
            )
            .group_by(MediaDB.type)
        )
        result = self.db.execute(annotated_images_frames_stmt)
        annotated_counts: dict[str, Any] = {f"annotated_{row.type}s": row.count for row in result}

        # Annotation Counts (annotated videos)
        annotated_video_stmt = (
            select(func.count(func.distinct(MediaDB.video_id)).label("count"))
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.annotation_data.isnot(None),
                DatasetItemDB.user_reviewed,
                MediaDB.type == "video_frame",
            )
        )
        annotated_video_count = self.db.execute(annotated_video_stmt).scalar()
        annotated_counts["annotated_videos"] = annotated_video_count or 0

        # Total instances:
        annotated_dataset_items_stmt = select(DatasetItemDB.annotation_data).where(
            DatasetItemDB.project_id == self.project_id,
            DatasetItemDB.annotation_data.isnot(None),
            DatasetItemDB.user_reviewed,
        )
        annotated_counts["instances"] = sum(
            len(item.annotation_data)
            for item in self.db.execute(annotated_dataset_items_stmt)
            if item.annotation_data is not None
        )

        # instances_per_label
        annotated_dataset_items_stmt = select(DatasetItemDB.annotation_data).where(
            DatasetItemDB.project_id == self.project_id, DatasetItemDB.annotation_data.isnot(None)
        )

        labels_counts = Counter(
            label["id"]
            for item in self.db.execute(annotated_dataset_items_stmt)
            if item.annotation_data is not None
            for annotation in item.annotation_data
            for label in annotation["labels"]
        )
        annotated_counts["instances_per_label"] = [
            {"label_id": label_id, "instances": count} for label_id, count in labels_counts.items()
        ]

        statistics.update(annotated_counts)

        return statistics
