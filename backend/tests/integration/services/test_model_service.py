# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi import UploadFile

from app.db.schema import ModelDB
from app.schemas import Model
from app.schemas.model_activation import ModelActivationState
from app.services import ResourceType
from app.services.model_service import ModelAlreadyExistsError, ModelService, ResourceNotFoundError


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
def fxt_upload_file():
    """Create a mock UploadFile for testing."""
    return UploadFile(
        file=io.BytesIO(b"some_content"),
        filename="text.txt",
    )


@pytest.fixture
def fxt_model_service(fxt_db_projects, db_session):
    """Fixture to create a ModelService instance with a temporary models directory."""
    db_project = fxt_db_projects[0]
    db_pipeline = db_project.pipeline
    db_pipeline.is_running = True
    db_session.add(db_project)
    db_session.flush()
    with TemporaryDirectory(suffix="models") as tmpdir:
        service = ModelService()
        service.models_dir = Path(tmpdir)
        service._model_activation_state = ModelActivationState(
            active_model=None, active_model_id=None, available_models=[]
        )
        return service


def create_model_db(db_session, models: list[ModelDB]) -> None:
    """Create a model in the database."""
    for db_model in models:
        db_session.add(db_model)
    db_session.flush()


def assert_model(actual: Model, expected: Model | ModelDB) -> None:
    assert str(actual.id) == expected.id
    assert actual.name == expected.name
    assert actual.format == expected.format


class TestModelServiceIntegration:
    """Integration tests for ModelService."""

    @pytest.mark.asyncio
    async def test_add_model_success(self, fxt_upload_file, fxt_model_service):
        """Test successfully adding a new model."""
        model_name = "new_detection_model"

        await fxt_model_service.add_model(model_name, fxt_upload_file, fxt_upload_file)

        assert model_name in fxt_model_service.get_available_model_names()
        assert fxt_model_service.get_active_model_name() == "new_detection_model"
        assert (fxt_model_service.models_dir / f"{model_name}.bin").exists()
        assert (fxt_model_service.models_dir / f"{model_name}.xml").exists()

    @pytest.mark.asyncio
    async def test_add_model_already_exists_error(self, fxt_upload_file, fxt_model_service):
        """Test adding a model that already exists raises error."""
        model_name = "card-detection-ssd"

        await fxt_model_service.add_model(model_name, fxt_upload_file, fxt_upload_file)

        with pytest.raises(ModelAlreadyExistsError, match=f"A model with the name '{model_name}' already exists"):
            await fxt_model_service.add_model(model_name, fxt_upload_file, fxt_upload_file)

    @pytest.mark.asyncio
    async def test_remove_model_success(self, fxt_upload_file, fxt_model_service):
        """Test successfully removing a model."""
        model_name = "remove_model"
        await fxt_model_service.add_model("model1", fxt_upload_file, fxt_upload_file)
        await fxt_model_service.add_model(model_name, fxt_upload_file, fxt_upload_file)

        fxt_model_service.remove_model(model_name)
        assert model_name not in fxt_model_service.get_available_model_names()
        assert not (fxt_model_service.models_dir / f"{model_name}.bin").exists()
        assert not (fxt_model_service.models_dir / f"{model_name}.xml").exists()

    def test_remove_model_not_found_error(self, fxt_model_service):
        """Test removing a non-existent model raises error."""
        model_name = "non_existent_model"

        with pytest.raises(ResourceNotFoundError):
            fxt_model_service.remove_model(model_name)

    @pytest.mark.asyncio
    async def test_activate_model_success(self, fxt_upload_file, fxt_model_service):
        """Test successfully activating an available model."""
        await fxt_model_service.add_model("model1", fxt_upload_file, fxt_upload_file)
        await fxt_model_service.add_model("model2", fxt_upload_file, fxt_upload_file)

        assert fxt_model_service.get_active_model_name() == "model1"

        fxt_model_service.activate_model("model2")

        assert fxt_model_service.get_active_model_name() == "model2"

    def test_activate_model_not_found_error(self, fxt_model_service):
        """Test activating a non-existent model raises error."""
        model_name = "non_existent_model"

        with pytest.raises(ResourceNotFoundError):
            fxt_model_service.activate_model(model_name)

    def test_list_models(self, fxt_db_models, fxt_model_service, db_session):
        """Test retrieving all models."""
        create_model_db(db_session, fxt_db_models)

        models = fxt_model_service.list_models()

        assert len(models) == len(fxt_db_models)
        for i, model in enumerate(models):
            assert_model(model, fxt_db_models[i])

    def test_get_model(self, fxt_db_models, fxt_model_service, db_session):
        """Test retrieving a model by ID."""
        db_model = fxt_db_models[0]
        create_model_db(db_session, [db_model])

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

    def test_delete_model(self, fxt_db_models, fxt_model_service, db_session):
        """Test deleting a model by ID."""
        db_model = fxt_db_models[0]
        create_model_db(db_session, models=[db_model])

        fxt_model_service.delete_model_by_id(UUID(db_model.id))

        assert db_session.query(ModelDB).count() == 0

    def test_delete_non_existent_model(self, fxt_model_service):
        """Test deleting a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.delete_model_by_id(model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

    def test_update_model(self, fxt_db_models, fxt_model_service, db_session):
        """Test deleting a model by ID."""
        db_model = fxt_db_models[0]
        create_model_db(db_session, models=[db_model])

        updated = fxt_model_service.update_model(UUID(db_model.id), {"name": "Updated Model Name"})
        db_updated = db_session.query(ModelDB).filter_by(id=db_model.id).one()

        assert_model(updated, db_updated)

    def test_update_non_existent_model(self, fxt_model_service):
        """Test deleting a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.update_model(model_id, {"name": "Updated Model Name"})

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)
