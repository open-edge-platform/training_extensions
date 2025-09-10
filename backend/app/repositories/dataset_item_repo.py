# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB
from app.repositories.base import BaseRepository


class DatasetItemRepository(BaseRepository[DatasetItemDB]):
    """Repository for dataset item-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, DatasetItemDB)
