from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import PipelineDB
from app.schemas.project import Label, Project, Task
from app.services.base import ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.project_service import ProjectService


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
def fxt_project() -> Project:
    """Fixture to create a Project instance."""
    labels = [Label(name="cat"), Label(name="dog")]
    task = Task(
        task_type="classification",
        exclusive_labels=True,
        labels=labels,
    )
    return Project(name="Test Project", task=task)


@pytest.fixture
def fxt_project_service() -> ProjectService:
    """Fixture to create a ProjectService instance."""
    return ProjectService()


class TestProjectServiceIntegration:
    """Integration tests for ProjectService."""

    def test_create_project(self, fxt_project_service: ProjectService, fxt_project: Project, db_session: Session):
        """Test creating a project."""
        project = fxt_project_service.create_project(fxt_project)
        assert project.id is not None
        assert project.name == "Test Project"
        assert project.task.task_type == "classification"
        assert len(project.task.labels) == 2
        # Ensure the associated pipeline is created
        db_pipeline = db_session.get(PipelineDB, str(project.id))
        assert db_pipeline is not None
        assert not db_pipeline.is_running

    def test_list_projects(self, fxt_project_service: ProjectService, fxt_project: Project):
        """Test listing projects."""
        # Create projects first
        for project in [fxt_project] * 3:
            project.id = uuid4()
            fxt_project_service.create_project(project)

        projects = fxt_project_service.list_projects()
        assert len(projects) == 3
        for i in range(3):
            assert projects[i].name == fxt_project.name
            assert projects[i].task.task_type == fxt_project.task.task_type
            assert projects[i].task.labels == fxt_project.task.labels

    def test_get_project_by_id(self, fxt_project_service: ProjectService, fxt_project: Project):
        """Test retrieving a project by ID."""
        created_project = fxt_project_service.create_project(fxt_project)
        fetched_project = fxt_project_service.get_project_by_id(created_project.id)
        assert fetched_project.id == created_project.id
        assert fetched_project.name == created_project.name

    def test_get_project_by_id_not_found(self, fxt_project_service: ProjectService):
        """Test retrieving a non-existent project raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.get_project_by_id(non_existent_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_project(self, fxt_project_service: ProjectService, fxt_project: Project, db_session: Session):
        """Test deleting a project."""
        created_project = fxt_project_service.create_project(fxt_project)
        fxt_project_service.delete_project_by_id(created_project.id)
        with pytest.raises(ResourceNotFoundError):
            fxt_project_service.get_project_by_id(created_project.id)

        # Ensure the associated pipeline is deleted
        assert db_session.get(PipelineDB, str(fxt_project.id)) is None

    def test_delete_active_project(
        self,
        fxt_project_service: ProjectService,
        fxt_project: Project,
        fxt_db_sources,
        fxt_db_sinks,
        fxt_db_models,
        db_session: Session,
    ):
        """Test deleting a project which has a running pipeline."""
        created_project = fxt_project_service.create_project(fxt_project)
        # Activate the associated pipeline after project is created
        db_pipeline = db_session.get(PipelineDB, str(created_project.id))
        assert db_pipeline is not None
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_models[0])
        db_pipeline.source_id = fxt_db_sources[0].id
        db_pipeline.sink_id = fxt_db_sinks[0].id
        db_pipeline.model_id = fxt_db_models[0].id
        db_pipeline.is_running = True
        db_session.flush()

        with pytest.raises(ResourceInUseError) as excinfo:
            fxt_project_service.delete_project_by_id(created_project.id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(created_project.id)
        assert "Cannot delete a project with a running pipeline." in str(excinfo.value)
