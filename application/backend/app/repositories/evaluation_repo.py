# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import EvaluationDB

from .base import BaseRepository


class EvaluationRepository(BaseRepository[EvaluationDB]):
    """Repository for evaluation-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, EvaluationDB)
