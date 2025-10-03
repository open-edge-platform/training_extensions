# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.db.schema import ProjectDB
from app.schemas import ProjectCreate, ProjectView
from app.schemas.project import Task, TaskType
from app.services.mappers.label_mapper import LabelMapper


class ProjectMapper:
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_db: ProjectDB) -> ProjectView:
        """Convert Project db entity to schema."""
        return ProjectView(
            id=UUID(project_db.id),
            active_pipeline=project_db.pipeline.is_running,
            name=project_db.name,
            task=Task(
                task_type=TaskType(project_db.task_type),
                exclusive_labels=project_db.exclusive_labels,
                labels=[LabelMapper.to_schema(db_label) for db_label in project_db.labels],
            ),
        )

    @staticmethod
    def from_schema(project: ProjectCreate) -> ProjectDB:
        """Convert Project schema to db model."""

        project_db = ProjectDB(
            id=str(project.id),
            name=project.name,
            task_type=project.task.task_type,
            exclusive_labels=project.task.exclusive_labels,
        )
        project_db.labels = [LabelMapper.from_schema(label_schema) for label_schema in project.task.labels]

        return project_db
