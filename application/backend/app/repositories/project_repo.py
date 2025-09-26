# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from app.db.schema import PipelineDB, ProjectDB
from app.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[ProjectDB]):
    """Repository for project-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ProjectDB)

    def save(self, project: ProjectDB) -> ProjectDB:
        # When a new project is created, also create an associated pipeline
        project.pipeline = PipelineDB(
            project_id=project.id,
        )
        return super().save(project)
