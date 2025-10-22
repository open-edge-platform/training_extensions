# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from model_api.models import Model

from app.schemas.model_activation import ModelActivationState
from app.services import ActiveModelService
from app.services.active_model_service import LoadedModel


@pytest.fixture
def fxt_model_activation_state() -> ModelActivationState:
    """Fixture to create a default ModelActivationState."""
    project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
    active_model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
    other_model_id = UUID("da21744f-990c-4b95-aa2a-70da8d46fdcf")
    available_models = [active_model_id, other_model_id]
    return ModelActivationState(
        project_id=project_id,
        active_model_id=active_model_id,
        available_models=available_models,
    )


@pytest.fixture
def fxt_active_model_service(fxt_model_activation_state, fxt_condition) -> Iterator[ActiveModelService]:
    """Fixture to create an ActiveModelService instance with mocked dependencies and a temporary data directory."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(ActiveModelService, "_load_state", return_value=fxt_model_activation_state),
    ):
        yield ActiveModelService(data_dir=Path(tmpdir), mp_model_reload_event=fxt_condition)


class TestActiveModelServiceUnit:
    """Unit tests for ActiveModelService."""

    def test_get_model_file_path(self, fxt_active_model_service):
        """Test retrieval of model file path."""
        project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
        model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
        extension = "bin"
        expected_path = fxt_active_model_service.projects_dir / f"{project_id}/models/{model_id}/model.{extension}"
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.touch()

        file_path = fxt_active_model_service._get_model_file_path(
            project_id=project_id, model_id=model_id, extension=extension
        )

        assert file_path == expected_path

    def test_get_model_file_path_not_found(self, fxt_active_model_service):
        """Test error when model file is not found."""
        project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
        model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
        extension = "bin"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            fxt_active_model_service._get_model_file_path(project_id=project_id, model_id=model_id, extension=extension)

    def test_get_loaded_inference_model(self, fxt_active_model_service, fxt_model_activation_state, monkeypatch):
        """Test loading the active inference model."""
        dummy_model = Mock(spec=Model)

        with (
            patch.object(Model, "create_model", return_value=dummy_model),
            patch.object(
                fxt_active_model_service,
                "_get_model_file_path",
                new=lambda project_id, model_id, ext: f"model.{ext}",
            ),
        ):
            loaded = fxt_active_model_service.get_loaded_inference_model(force_reload=True)
            assert isinstance(loaded, LoadedModel)
            assert loaded.id == fxt_model_activation_state.active_model_id
            assert loaded.model is dummy_model
