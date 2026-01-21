# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import EvaluationDB

from .base import BaseRepository


class EvaluationRepository(BaseRepository[EvaluationDB]):
    """Repository for evaluation-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, EvaluationDB)

    def list_by_model_id(self, model_id: str) -> Sequence[EvaluationDB]:
        stmt = select(EvaluationDB).where(EvaluationDB.model_revision_id == model_id)
        return self.db.execute(stmt).scalars().all()
