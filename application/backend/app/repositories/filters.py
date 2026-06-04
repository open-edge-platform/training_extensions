# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import Select, exists, func, select
from sqlalchemy.orm import aliased

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB
from app.models import DatasetItemAnnotationStatus, MediaType


def _apply_annotation_status_filter(stmt: Select, annotation_status: str | None = None) -> Select:
    """Apply annotation status filter to a select statement."""
    match annotation_status:
        case DatasetItemAnnotationStatus.MISSING_ANNOTATIONS:
            stmt = stmt.where(DatasetItemDB.annotation_data.is_(None))
        case DatasetItemAnnotationStatus.WITH_ANNOTATIONS:
            stmt = stmt.where(DatasetItemDB.annotation_data.isnot(None))
    return stmt


def _apply_annotation_status_filter_with_video_support(stmt: Select, annotation_status: str | None = None) -> Select:
    """Apply annotation status filter to a media query, accounting for the video/frame hierarchy.

    Unlike the base filter, this handles the case where videos do not have their own
    ``DatasetItemDB`` rows. A video is considered annotated if at least one of its
    frames has a ``DatasetItemDB`` with non-null ``annotation_data``.

    Args:
        stmt: The base SQLAlchemy select statement, expected to have ``MediaDB`` as its
              primary entity with an outer join to ``DatasetItemDB``.
        annotation_status: One of ``DatasetItemAnnotationStatus`` values, or ``None``
                           to skip filtering.

    Returns:
        The select statement with the annotation status condition applied.
    """
    if annotation_status == DatasetItemAnnotationStatus.WITH_ANNOTATIONS:
        # Images/frames: have a dataset_item with annotation_data
        # Videos: have no dataset_item themselves, but at least one annotated frame
        frame_alias = aliased(MediaDB)
        annotated_frame_exists = exists(
            select(DatasetItemDB.id)
            .join(frame_alias, frame_alias.id == DatasetItemDB.id)
            .where(
                frame_alias.video_id == MediaDB.id,
                DatasetItemDB.annotation_data.isnot(None),
            )
            .correlate(MediaDB)
        )
        stmt = stmt.where(
            (MediaDB.type != MediaType.VIDEO)
            & (DatasetItemDB.annotation_data.isnot(None))  # image or frame with annotation
            | (
                (MediaDB.type == MediaType.VIDEO) & annotated_frame_exists  # video with at least one annotated frame
            )
        )
    elif annotation_status == DatasetItemAnnotationStatus.MISSING_ANNOTATIONS:
        frame_alias = aliased(MediaDB)
        annotated_frame_count = (
            select(func.count(DatasetItemDB.id))
            .join(frame_alias, frame_alias.id == DatasetItemDB.id)
            .where(
                frame_alias.video_id == MediaDB.id,
                DatasetItemDB.annotation_data.isnot(None),
            )
            .correlate(MediaDB)
            .scalar_subquery()
        )
        stmt = stmt.where(
            (MediaDB.type != MediaType.VIDEO)
            & (DatasetItemDB.annotation_data.is_(None))  # image or frame missing annotation
            | (
                (MediaDB.type == MediaType.VIDEO)
                & (annotated_frame_count < MediaDB.frame_count)  # not all frames annotated
            )
        )
    return stmt


def _apply_label_filter_with_video_support(stmt: Select, label_ids: list[str] | None = None) -> Select:
    """Apply a label filter to a media query, accounting for the video/frame hierarchy.

    Unlike a plain join on ``DatasetItemLabelDB``, this handles the case where videos do
    not have their own ``DatasetItemDB`` rows. A video is considered to match a label if
    at least one of its frames has a ``DatasetItemDB`` annotated with one of the given
    labels. In that case the video itself is returned (not its individual frames).

    Images and video frames match when their own dataset item carries one of the labels.

    Args:
        stmt: The base SQLAlchemy select statement, expected to have ``MediaDB`` as its
              primary entity.
        label_ids: The list of label ids to filter by, or ``None``/empty to skip filtering.

    Returns:
        The select statement with the label condition applied.
    """
    if not label_ids:
        return stmt

    # Image or frame: its own dataset item carries one of the requested labels
    own_label_exists = exists(
        select(DatasetItemLabelDB.dataset_item_id)
        .where(
            DatasetItemLabelDB.dataset_item_id == MediaDB.id,
            DatasetItemLabelDB.label_id.in_(label_ids),
        )
        .correlate(MediaDB)
    )

    # Video: at least one of its frames has a dataset item carrying one of the requested labels
    frame_alias = aliased(MediaDB)
    frame_label_exists = exists(
        select(DatasetItemLabelDB.dataset_item_id)
        .join(frame_alias, frame_alias.id == DatasetItemLabelDB.dataset_item_id)
        .where(
            frame_alias.video_id == MediaDB.id,
            DatasetItemLabelDB.label_id.in_(label_ids),
        )
        .correlate(MediaDB)
    )

    return stmt.where(
        ((MediaDB.type != MediaType.VIDEO) & own_label_exists)
        | ((MediaDB.type == MediaType.VIDEO) & frame_label_exists)
    )


def _apply_subset_filter(stmt: Select, subset: str | None = None) -> Select:
    """Apply subset filter to a select statement."""
    if subset is not None:
        stmt = stmt.where(DatasetItemDB.subset == subset)
    return stmt
