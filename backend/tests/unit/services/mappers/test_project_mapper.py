# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.db.schema import LabelDB, ProjectDB
from app.schemas.project import Label, Project, Task, TaskType
from app.services.mappers import LabelMapper, ProjectMapper

SUPPORTED_PROJECT_MAPPING = [
    (
        Project(
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
        Project(name="Test Project", task=Task(task_type=TaskType.DETECTION, exclusive_labels=False, labels=[])),
        ProjectDB(name="Test Project", task_type=TaskType.DETECTION, exclusive_labels=False),
    ),
]


class TestProjectMapper:
    """Test suite for ProjectMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", SUPPORTED_PROJECT_MAPPING.copy())
    def test_from_schema(self, schema_instance, expected_db):
        expected_db.id = str(schema_instance.id)
        expected_db.labels = [LabelMapper.from_schema(schema_label) for schema_label in schema_instance.task.labels]
        actual_db = ProjectMapper.from_schema(schema_instance)
        assert actual_db.id == expected_db.id
        assert actual_db.name == expected_db.name
        assert actual_db.task_type == expected_db.task_type
        assert actual_db.exclusive_labels == expected_db.exclusive_labels
        assert {label.name for label in actual_db.labels} == {label.name for label in expected_db.labels}

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_PROJECT_MAPPING.copy()])
    def test_to_schema(self, db_instance, expected_schema):
        db_instance.id = str(expected_schema.id)
        db_instance.labels = [
            LabelDB(id=str(schema_label.id), name=schema_label.name) for schema_label in expected_schema.task.labels
        ]
        actual_schema = ProjectMapper.to_schema(db_instance)
        assert actual_schema.name == expected_schema.name
        assert actual_schema.task.task_type == expected_schema.task.task_type
        assert actual_schema.task.exclusive_labels == expected_schema.task.exclusive_labels
        assert {label.name for label in actual_schema.task.labels} == {
            label.name for label in expected_schema.task.labels
        }
