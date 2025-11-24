# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.db.schema import ProjectDB
from app.models import Label
from app.schemas.label import LabelCreate, LabelView
from app.schemas.project import ProjectCreate, ProjectView, TaskCreate, TaskType, TaskView
from app.services.mappers import ProjectMapper

CREATE_PROJECT_TO_DB_MAPPING = [
    (
        ProjectCreate(
            name="Test Project",
            task=TaskCreate(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
                labels=[
                    LabelCreate(name="label1", color="#FF0000", hotkey=None),
                    LabelCreate(name="label2", color="#00FF00", hotkey=None),
                ],
            ),
        ),
        ProjectDB(name="Test Project", task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
    ),
    (
        ProjectCreate(
            name="Test Project", task=TaskCreate(task_type=TaskType.DETECTION, exclusive_labels=False, labels=[])
        ),
        ProjectDB(name="Test Project", task_type=TaskType.DETECTION, exclusive_labels=False),
    ),
]

PROJECT_ID = uuid4()
LABEL1_ID = uuid4()
LABEL2_ID = uuid4()
DB_TO_VIEW_PROJECT_MAPPING = [
    (
        ProjectDB(id=str(PROJECT_ID), name="Test Project", task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
        ProjectView(
            id=PROJECT_ID,
            name="Test Project",
            active_pipeline=True,
            task=TaskView(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
                labels=[
                    LabelView(id=LABEL1_ID, name="label1", color="#FF0000", hotkey=None),
                    LabelView(id=LABEL2_ID, name="label2", color="#00FF00", hotkey=None),
                ],
            ),
        ),
        [
            Label(id=LABEL1_ID, project_id=PROJECT_ID, name="label1", color="#FF0000", hotkey=None),
            Label(id=LABEL2_ID, project_id=PROJECT_ID, name="label2", color="#00FF00", hotkey=None),
        ],
    ),
    (
        ProjectDB(id=str(PROJECT_ID), name="Test Project", task_type=TaskType.DETECTION, exclusive_labels=False),
        ProjectView(
            id=PROJECT_ID,
            name="Test Project",
            active_pipeline=False,
            task=TaskView(task_type=TaskType.DETECTION, exclusive_labels=False, labels=[]),
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
    @pytest.mark.parametrize("active_pipeline", [True, False])
    def test_to_schema(
        self, db_instance: ProjectDB, expected_schema: ProjectView, labels: list[Label], active_pipeline
    ) -> None:
        db_instance.id = str(expected_schema.id)
        actual_schema = ProjectMapper.to_schema(db_instance, active_pipeline, labels)
        assert actual_schema.name == expected_schema.name
        assert actual_schema.task.task_type == expected_schema.task.task_type
        assert actual_schema.task.exclusive_labels == expected_schema.task.exclusive_labels
        assert actual_schema.active_pipeline == active_pipeline
        assert {label.name for label in actual_schema.task.labels} == {
            label.name for label in expected_schema.task.labels
        }
