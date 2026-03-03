# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.orm import Session

from app.db.schema import ModelVariantDB
from app.repositories.base import BaseRepository


class ModelVariantRepository(BaseRepository[ModelVariantDB]):
    """Repository for model variant database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ModelVariantDB)

    def list_by_model_revision(self, model_revision_id: str) -> Sequence[ModelVariantDB]:
        """List all variants for a given model revision."""
        stmt = select(ModelVariantDB).where(ModelVariantDB.model_revision_id == model_revision_id)
        return self.db.execute(stmt).scalars().all()

    def get_by_id(self, obj_id: str) -> ModelVariantDB | None:
        """Get a model variant by its ID."""
        stmt = select(ModelVariantDB).where(ModelVariantDB.id == obj_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_revision_format_precision(
        self, model_revision_id: str, format: str, precision: str
    ) -> ModelVariantDB | None:
        """Get a model variant by revision, format, and precision."""
        stmt = select(ModelVariantDB).where(
            (ModelVariantDB.model_revision_id == model_revision_id)
            & (ModelVariantDB.format == format)
            & (ModelVariantDB.precision == precision)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def delete(self, obj_id: str) -> bool:
        """Delete a model variant by its ID."""
        stmt = delete(ModelVariantDB).where(ModelVariantDB.id == obj_id)
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

    def mark_files_deleted(self, variant_id: str) -> None:
        """Mark a variant's files as deleted."""
        stmt = update(ModelVariantDB).where(ModelVariantDB.id == variant_id).values(files_deleted=True)
        self.db.execute(stmt)
