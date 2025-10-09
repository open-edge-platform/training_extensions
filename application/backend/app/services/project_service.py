# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import DatasetItemRepository, PipelineRepository, ProjectRepository
from app.repositories.base import PrimaryKeyIntegrityError
from app.schemas import Label, Project
from app.services.base import ResourceInUseError, ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError
from app.services.label_service import LabelService
from app.services.mappers.project_mapper import ProjectMapper
from app.services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_ACTIVE_PROJECT = "Cannot delete a project with a running pipeline."


class ProjectService:
    def __init__(self, data_dir: Path, db_session: Session, label_service: LabelService) -> None:
        self._projects_dir = data_dir / "projects"
        self._db_session: Session = db_session
        self._label_service: LabelService = label_service

    @parent_process_only
    def create_project(self, project: Project) -> Project:
        project_repo = ProjectRepository(self._db_session)
        try:
            saved = project_repo.save(ProjectMapper.from_schema(project))
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.PROJECT, str(project.id))
        labels: list[Label] = []
        for label in project.task.labels:
            labels.append(self._label_service.create_label(project_id=UUID(saved.id), label=label))
        return ProjectMapper.to_schema(saved, labels)

    def list_projects(self) -> list[Project]:
        project_repo = ProjectRepository(self._db_session)
        projects = project_repo.list_all()
        result: list[Project] = []
        for project in projects:
            labels = self._label_service.list_all(project_id=UUID(project.id))
            result.append(ProjectMapper.to_schema(project, labels))
        return result

    def get_project_by_id(self, project_id: UUID) -> Project:
        project_repo = ProjectRepository(self._db_session)
        project_db = project_repo.get_by_id(str(project_id))
        if not project_db:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        return ProjectMapper.to_schema(project_db, self._label_service.list_all(project_id))

    @parent_process_only
    def update_project_name(self, project_id: UUID, name: str) -> Project:
        """Update only the project name"""
        project_repo = ProjectRepository(self._db_session)
        project_db = project_repo.get_by_id(str(project_id))
        if not project_db:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        project_db.name = name
        return ProjectMapper.to_schema(project_repo.update(project_db), self._label_service.list_all(project_id))

    @parent_process_only
    def delete_project_by_id(self, project_id: UUID) -> None:
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline_db = pipeline_repo.get_by_id(str(project_id))
        if pipeline_db and pipeline_db.is_running:
            raise ResourceInUseError(ResourceType.PROJECT, str(project_id), MSG_ERR_DELETE_ACTIVE_PROJECT)
        project_repo = ProjectRepository(self._db_session)
        if not project_repo.delete(str(project_id)):
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

    def get_project_thumbnail_path(self, project_id: UUID) -> Path | None:
        """Get the path to the project's thumbnail image, as determined by the earliest dataset item"""
        dataset_item_repo = DatasetItemRepository(project_id=str(project_id), db=self._db_session)
        earliest_dataset_item = dataset_item_repo.get_earliest()

        if earliest_dataset_item:
            return self._projects_dir / f"{project_id}/dataset/{earliest_dataset_item.id}-thumb.jpg"
        return None
