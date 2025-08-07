import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi import UploadFile

from app.schemas.model_activation import ModelActivationState
from app.services.model_service import ModelAlreadyExistsError, ModelNotFoundError, ModelService


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.model_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock


@pytest.fixture
def fxt_upload_file():
    """Create a mock UploadFile for testing."""
    return UploadFile(
        file=io.BytesIO(b"some_content"),
        filename="text.txt",
    )


@pytest.fixture
def fxt_model_service(fxt_default_pipeline):
    with TemporaryDirectory(suffix="models") as tmpdir:
        service = ModelService()
        service.models_dir = Path(tmpdir)
        service._model_activation_state = ModelActivationState(active_model=None, available_models=[])
        yield service


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
    async def test_remove_model_success(self, fxt_upload_file):
        """Test successfully removing a model."""
        model_name = "remove_model"
        model_service = ModelService()
        await model_service.add_model(model_name, fxt_upload_file, fxt_upload_file)

        model_service.remove_model(model_name)
        assert model_name not in model_service.get_available_model_names()
        assert not (model_service.models_dir / f"{model_name}.bin").exists()
        assert not (model_service.models_dir / f"{model_name}.xml").exists()

    def test_remove_model_not_found_error(self):
        """Test removing a non-existent model raises error."""
        model_name = "non_existent_model"
        model_service = ModelService()

        with pytest.raises(ModelNotFoundError, match="Model 'non_existent_model' not found"):
            model_service.remove_model(model_name)

    @pytest.mark.asyncio
    async def test_remove_active_model_activates_next(self, fxt_upload_file, fxt_model_service):
        """Test removing active model activates the next available model."""
        await fxt_model_service.add_model("model1", fxt_upload_file, fxt_upload_file)
        await fxt_model_service.add_model("model2", fxt_upload_file, fxt_upload_file)

        available_models = fxt_model_service.get_available_model_names()
        assert len(available_models) == 2
        assert "model1" in available_models
        assert "model2" in available_models
        assert fxt_model_service.get_active_model_name() == "model1"

        fxt_model_service.remove_model("model1")

        assert fxt_model_service.get_active_model_name() == "model2"
        assert len(available_models) == 1
        assert "model1" not in fxt_model_service.get_available_model_names()
        assert "model2" in fxt_model_service.get_available_model_names()

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

        with pytest.raises(ModelNotFoundError, match=f"Model '{model_name}' not found"):
            fxt_model_service.activate_model(model_name)
