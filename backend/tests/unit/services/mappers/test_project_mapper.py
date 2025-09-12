# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from app.db.schema import ProjectDB
from app.schemas.project import Label, Project, Task
from app.services.mappers import ProjectMapper

SUPPORTED_PROJECT_MAPPING = [
    (
        Project(
            name="Test Project",
            task=Task(
                task_type="classification", exclusive_labels=True, labels=[Label(name="label1"), Label(name="label2")]
            ),
        ),
        ProjectDB(name="Test Project", task_type="classification", exclusive_labels=True, labels=["label1", "label2"]),
    ),
    (
        Project(name="Test Project", task=Task(task_type="detection", exclusive_labels=False, labels=[])),
        ProjectDB(name="Test Project", task_type="detection", exclusive_labels=False, labels=[]),
    ),
]


class TestProjectMapper:
    """Test suite for ProjectMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", SUPPORTED_PROJECT_MAPPING)
    def test_from_schema(self, schema_instance, expected_db):
        expected_db.id = str(schema_instance.id)
        actual_db = ProjectMapper.from_schema(schema_instance)
        assert actual_db.id == expected_db.id
        assert actual_db.name == expected_db.name
        assert actual_db.task_type == expected_db.task_type
        assert actual_db.exclusive_labels == expected_db.exclusive_labels
        assert actual_db.labels == expected_db.labels

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_PROJECT_MAPPING])
    def test_to_schema(self, db_instance, expected_schema):
        db_instance.id = str(expected_schema.id)
        actual_schema = ProjectMapper.to_schema(db_instance)
        assert actual_schema == expected_schema
