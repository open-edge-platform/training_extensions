# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from app.db import get_db_session
from app.repositories import DatasetItemRepository, PipelineRepository, ProjectRepository
from app.schemas import Project
from app.services.base import (
    GenericPersistenceService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ServiceConfig,
)
from app.services.mappers.project_mapper import ProjectMapper
from app.services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_ACTIVE_PROJECT = "Cannot delete a project with a running pipeline."


class ProjectService:
    def __init__(self, data_dir: Path) -> None:
        self._persistence: GenericPersistenceService[Project, ProjectRepository] = GenericPersistenceService(
            ServiceConfig(ProjectRepository, ProjectMapper, ResourceType.PROJECT)
        )
        self.projects_dir = data_dir / "projects"

    @parent_process_only
    def create_project(self, project: Project) -> Project:
        return self._persistence.create(project)

    def list_projects(self) -> list[Project]:
        return self._persistence.list_all()

    def get_project_by_id(self, project_id: UUID) -> Project:
        project = self._persistence.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        return project

    @parent_process_only
    def delete_project_by_id(self, project_id: UUID) -> None:
        with get_db_session() as db:
            pipeline_repo = PipelineRepository(db)
            pipeline_db = pipeline_repo.get_by_id(str(project_id))
            if pipeline_db and pipeline_db.is_running:
                raise ResourceInUseError(ResourceType.PROJECT, str(project_id), MSG_ERR_DELETE_ACTIVE_PROJECT)
            self._persistence.delete_by_id(project_id, db)
            db.commit()

    def get_project_thumbnail_path(self, project_id: UUID) -> Path | None:
        """Get the path to the project's thumbnail image, as determined by the earliest dataset item"""
        with get_db_session() as db:
            dataset_repo = DatasetItemRepository(str(project_id), db)
            earliest_dataset_item = dataset_repo.get_earliest()

        if earliest_dataset_item:
            return self.projects_dir / f"{project_id}/dataset/{earliest_dataset_item.id}-thumb.jpg"
        return None
