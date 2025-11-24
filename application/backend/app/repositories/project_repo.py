# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import ProjectDB
from app.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[ProjectDB]):
    """Repository for project-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ProjectDB)
