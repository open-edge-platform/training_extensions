# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import ModelRevisionDB, ProjectDB
from app.schemas import Model
from app.schemas.model_activation import ModelActivationState
from app.services import ModelService, ResourceNotFoundError, ResourceType


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with (
        patch("app.services.model_service.get_db_session") as mock,
        patch("app.services.base.get_db_session") as mock_base,
    ):
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        mock_base.return_value.__enter__.return_value = db_session
        mock_base.return_value.__exit__.return_value = None
        yield


@pytest.fixture
def fxt_model_service(fxt_db_projects, db_session):
    """Fixture to create a ModelService instance with a temporary models directory."""
    db_project = fxt_db_projects[0]
    db_pipeline = db_project.pipeline
    db_pipeline.is_running = True
    db_session.add(db_project)
    db_session.flush()
    with TemporaryDirectory(suffix="models") as tmpdir:
        service = ModelService(Path(tmpdir).parent)
        service._model_activation_state = ModelActivationState(active_model_id=None, available_models=[])
        return service


def create_model_db(project: ProjectDB, models: list[ModelRevisionDB], db_session) -> None:
    """Create a model in the database."""
    db_session.add(project)
    for db_model in models:
        db_model.project_id = project.id
        db_session.add(db_model)
    db_session.flush()


def assert_model(actual: Model, expected: Model | ModelRevisionDB) -> None:
    assert str(actual.id) == expected.id
    assert actual.architecture == expected.architecture


class TestModelServiceIntegration:
    """Integration tests for ModelService."""

    def test_list_models(self, fxt_db_projects, fxt_db_models, fxt_model_service, db_session):
        """Test retrieving all models."""
        create_model_db(fxt_db_projects[0], fxt_db_models, db_session)

        models = fxt_model_service.list_models()

        assert len(models) == len(fxt_db_models)
        for i, model in enumerate(models):
            assert_model(model, fxt_db_models[i])

    def test_get_model(self, fxt_db_projects, fxt_db_models, fxt_model_service, db_session):
        """Test retrieving a model by ID."""
        db_model = fxt_db_models[0]
        create_model_db(fxt_db_projects[0], [db_model], db_session)

        model = fxt_model_service.get_model_by_id(UUID(db_model.id))

        assert model is not None
        assert_model(model, db_model)

    def test_get_non_existent_model(self, fxt_model_service):
        """Test retrieving a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.get_model_by_id(model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

    def test_delete_model(self, fxt_db_projects, fxt_db_models, fxt_model_service, db_session):
        """Test deleting a model by ID."""
        db_model = fxt_db_models[0]
        create_model_db(fxt_db_projects[0], [db_model], db_session)

        fxt_model_service.delete_model_by_id(UUID(db_model.id))

        assert db_session.query(ModelRevisionDB).count() == 0

    def test_delete_non_existent_model(self, fxt_model_service):
        """Test deleting a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.delete_model_by_id(model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)
