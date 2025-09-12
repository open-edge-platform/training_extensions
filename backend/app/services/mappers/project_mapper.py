# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.db.schema import ProjectDB
from app.schemas import Project


class ProjectMapper:
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_db: ProjectDB) -> Project:
        """Convert Project db entity to schema."""
        task_dict = {
            "task_type": project_db.task_type,
            "exclusive_labels": project_db.exclusive_labels,
            "labels": [{"name": label} for label in project_db.labels],
        }
        return Project.model_validate({"id": project_db.id, "name": project_db.name, "task": task_dict})

    @staticmethod
    def from_schema(project: Project) -> ProjectDB:
        """Convert Project schema to db model."""

        return ProjectDB(
            id=str(project.id),
            name=project.name,
            task_type=project.task.task_type,
            exclusive_labels=project.task.exclusive_labels,
            labels=[label.name for label in project.task.labels],
        )
