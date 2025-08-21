# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import SourceDB
from app.repositories.base import BaseRepository


class SourceRepository(BaseRepository[SourceDB]):
    """Repository for source-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, SourceDB)
