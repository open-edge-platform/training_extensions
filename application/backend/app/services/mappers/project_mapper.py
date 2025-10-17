# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.db.schema import ProjectDB
from app.schemas import LabelView, ProjectCreate, ProjectView
from app.schemas.project import TaskType, TaskView


class ProjectMapper:
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_db: ProjectDB, labels: list[LabelView]) -> ProjectView:
        """Convert Project db entity to schema."""
        return ProjectView(
            id=UUID(project_db.id),
            active_pipeline=project_db.pipeline.is_running,
            name=project_db.name,
            task=TaskView(
                task_type=TaskType(project_db.task_type),
                exclusive_labels=project_db.exclusive_labels,
                labels=labels,
            ),
        )

    @staticmethod
    def from_schema(project: ProjectCreate) -> ProjectDB:
        """Convert Project schema to db model."""

        return ProjectDB(
            id=str(project.id),
            name=project.name,
            task_type=project.task.task_type,
            exclusive_labels=project.task.exclusive_labels,
        )
