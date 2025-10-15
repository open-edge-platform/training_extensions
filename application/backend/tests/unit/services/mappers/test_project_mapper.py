# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.db.schema import PipelineDB, ProjectDB
from app.schemas.project import Label, ProjectCreate, ProjectView, Task, TaskType
from app.services.mappers import ProjectMapper

CREATE_PROJECT_TO_DB_MAPPING = [
    (
        ProjectCreate(
            name="Test Project",
            task=Task(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
                labels=[Label(name="label1"), Label(name="label2")],  # type: ignore[call-arg]
            ),
        ),
        ProjectDB(name="Test Project", task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
    ),
    (
        ProjectCreate(name="Test Project", task=Task(task_type=TaskType.DETECTION, exclusive_labels=False, labels=[])),
        ProjectDB(name="Test Project", task_type=TaskType.DETECTION, exclusive_labels=False),
    ),
]

PROJECT_ID = uuid4()
DB_TO_VIEW_PROJECT_MAPPING = [
    (
        ProjectDB(id=str(PROJECT_ID), name="Test Project", task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
        ProjectView(
            id=PROJECT_ID,
            name="Test Project",
            active_pipeline=True,
            task=Task(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
                labels=[Label(name="label1"), Label(name="label2")],  # type: ignore[call-arg]
            ),
        ),
        [Label(name="label1"), Label(name="label2")],  # type: ignore[call-arg]
    ),
    (
        ProjectDB(id=str(PROJECT_ID), name="Test Project", task_type=TaskType.DETECTION, exclusive_labels=False),
        ProjectView(
            id=PROJECT_ID,
            name="Test Project",
            active_pipeline=False,
            task=Task(task_type=TaskType.DETECTION, exclusive_labels=False, labels=[]),
        ),
        [],
    ),
]


class TestProjectMapper:
    """Test suite for ProjectMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", CREATE_PROJECT_TO_DB_MAPPING.copy())
    def test_from_schema(self, schema_instance: ProjectCreate, expected_db: ProjectDB) -> None:
        actual_db = ProjectMapper.from_schema(schema_instance)
        assert actual_db.id == str(schema_instance.id)
        assert actual_db.name == expected_db.name
        assert actual_db.task_type == expected_db.task_type
        assert actual_db.exclusive_labels == expected_db.exclusive_labels

    @pytest.mark.parametrize("db_instance,expected_schema, labels", DB_TO_VIEW_PROJECT_MAPPING.copy())
    def test_to_schema(self, db_instance: ProjectDB, expected_schema: ProjectView, labels: list[Label]) -> None:
        db_instance.id = str(expected_schema.id)
        db_instance.pipeline = PipelineDB(is_running=expected_schema.active_pipeline or False)
        actual_schema = ProjectMapper.to_schema(db_instance, labels)
        assert actual_schema.name == expected_schema.name
        assert actual_schema.task.task_type == expected_schema.task.task_type
        assert actual_schema.task.exclusive_labels == expected_schema.task.exclusive_labels
        assert {label.name for label in actual_schema.task.labels} == {
            label.name for label in expected_schema.task.labels
        }
