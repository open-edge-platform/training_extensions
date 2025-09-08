# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.db import get_db_session
from app.repositories import PipelineRepository, ProjectRepository
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
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Project, ProjectRepository] = GenericPersistenceService(
            ServiceConfig(ProjectRepository, ProjectMapper, ResourceType.PROJECT)
        )

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
