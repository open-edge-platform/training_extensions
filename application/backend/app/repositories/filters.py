# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import Select

from app.db.schema import DatasetItemDB
from app.models import DatasetItemAnnotationStatus


def _apply_annotation_status_filter(stmt: Select, annotation_status: str | None = None) -> Select:
    """Apply annotation status filter to a select statement."""
    match annotation_status:
        case DatasetItemAnnotationStatus.UNANNOTATED:
            stmt = stmt.where(DatasetItemDB.annotation_data.is_(None))
        case DatasetItemAnnotationStatus.REVIEWED:
            stmt = stmt.where(DatasetItemDB.user_reviewed.is_(True))
        case DatasetItemAnnotationStatus.TO_REVIEW:
            stmt = stmt.where(DatasetItemDB.user_reviewed.is_(False))
        case DatasetItemAnnotationStatus.REVIEWED_WITH_UNANNOTATED:
            stmt = stmt.where(DatasetItemDB.annotation_data.is_(None) | DatasetItemDB.user_reviewed.is_(True))
    return stmt


def _apply_subset_filter(stmt: Select, subset: str | None = None) -> Select:
    """Apply subset filter to a select statement."""
    if subset is not None:
        stmt = stmt.where(DatasetItemDB.subset == subset)
    return stmt
