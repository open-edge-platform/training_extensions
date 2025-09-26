# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, LabelDB, PipelineDB, ProjectDB
from app.schemas.project import Label, Project, Task, TaskType
from app.services.base import ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.project_service import ProjectService


@pytest.fixture()
def fxt_projects_dir() -> Generator[Path]:
    """Setup a temporary data directory for tests."""
    projects_dir = Path("data/projects")
    if not projects_dir.exists():
        projects_dir.mkdir(parents=True)
    yield projects_dir
    shutil.rmtree(projects_dir)


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with (
        patch("app.services.project_service.get_db_session") as mock,
        patch("app.services.base.get_db_session") as mock_base,
    ):
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        mock_base.return_value.__enter__.return_value = db_session
        mock_base.return_value.__exit__.return_value = None
        yield


@pytest.fixture
def fxt_project_service(fxt_projects_dir: Path) -> ProjectService:
    """Fixture to create a ProjectService instance."""
    return ProjectService(fxt_projects_dir.parent)


class TestProjectServiceIntegration:
    """Integration tests for ProjectService."""

    def test_create_project(self, fxt_project_service: ProjectService, db_session: Session):
        """Test creating a project."""
        labels = [
            Label(name="cat", color="#00FF00", hotkey="c"),
            Label(name="dog", color="#FF0000", hotkey="d"),
        ]
        new_project = Project(
            name="Test Project",
            task=Task(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
                labels=labels,
            ),
        )
        created_project = fxt_project_service.create_project(new_project)
        assert created_project.id is not None
        assert created_project.name == new_project.name
        assert created_project.task.task_type == new_project.task.task_type
        assert created_project.task.labels == new_project.task.labels
        # Ensure the associated pipeline is created
        db_pipeline = db_session.get(PipelineDB, str(created_project.id))
        assert db_pipeline is not None
        assert not db_pipeline.is_running
        # Ensure labels are created
        db_labels = db_session.query(LabelDB).filter(LabelDB.project_id == str(created_project.id)).all()
        assert len(db_labels) == 2
        assert {label.name for label in db_labels} == {"cat", "dog"}

    def test_list_projects(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test listing projects."""
        # Create projects first
        for db_project in fxt_db_projects:
            db_session.add(db_project)
        db_session.flush()

        projects = fxt_project_service.list_projects()
        assert len(projects) == 3
        for i in range(3):
            assert projects[i].name == fxt_db_projects[i].name
            assert projects[i].task.task_type == fxt_db_projects[i].task_type
            assert {label.name for label in projects[i].task.labels} == {
                label.name for label in fxt_db_projects[i].labels
            }

    def test_get_project_by_id(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test retrieving a project by ID."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        fetched_project = fxt_project_service.get_project_by_id(UUID(db_project.id))
        assert str(fetched_project.id) == db_project.id
        assert fetched_project.name == db_project.name

    def test_get_project_thumbnail(self, fxt_project_service: ProjectService, db_session: Session):
        """Test retrieving a project returns correct thumbnail ID."""
        # First create a project
        db_project = ProjectDB(
            id=str(uuid4()),
            name="P1",
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
        )
        db_session.add(db_project)
        db_session.flush()

        # No dataset items yet, should return None
        fetched_thumbnail_path = fxt_project_service.get_project_thumbnail_path(UUID(db_project.id))
        assert fetched_thumbnail_path is None

        # Add a dataset item
        db_dataset_item = DatasetItemDB(
            id=str(uuid4()),
            project_id=db_project.id,
            name="item1",
            format="jpg",
            width=1920,
            height=1080,
            size=1024,
            subset="unassigned",
        )
        db_session.add(db_dataset_item)
        db_session.flush()

        # Now it should return the path to the thumbnail
        fetched_thumbnail_path = fxt_project_service.get_project_thumbnail_path(UUID(db_project.id))
        assert (
            fetched_thumbnail_path
            == fxt_project_service.projects_dir / f"{db_project.id}/dataset/{db_dataset_item.id}-thumb.jpg"
        )

    def test_get_project_by_id_not_found(self, fxt_project_service: ProjectService):
        """Test retrieving a non-existent project raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.get_project_by_id(non_existent_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_project(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test deleting a project."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        fxt_project_service.delete_project_by_id(UUID(db_project.id))

        assert db_session.get(ProjectDB, db_project.id) is None
        # Ensure the associated pipeline is deleted
        assert db_session.get(PipelineDB, str(db_project.id)) is None

    def test_delete_active_project(
        self,
        fxt_project_service: ProjectService,
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
    ):
        """Test deleting a project which has a running pipeline."""
        db_project = fxt_db_projects[0]
        db_pipeline = db_project.pipeline
        db_pipeline.is_running = True
        db_session.add(db_project)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as excinfo:
            fxt_project_service.delete_project_by_id(UUID(db_project.id))

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == db_project.id
        assert "Cannot delete a project with a running pipeline." in str(excinfo.value)

    def test_delete_project_not_found(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test deleting a non-existing project."""
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.delete_project_by_id(non_existent_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)
