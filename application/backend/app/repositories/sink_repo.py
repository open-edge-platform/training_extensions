# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import SinkDB
from app.repositories.base import BaseRepository


class SinkRepository(BaseRepository[SinkDB]):
    """Repository for sink-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, SinkDB)
