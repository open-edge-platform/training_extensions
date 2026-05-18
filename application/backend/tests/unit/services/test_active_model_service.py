# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from model_api.models import Model

from app.models.model_activation import ModelActivationState
from app.models.system import DeviceInfo, DeviceType
from app.services import ActiveModelService
from app.services.inference.model_loader import LoadedModelHandle


@pytest.fixture
def fxt_model_activation_state() -> ModelActivationState:
    """Fixture to create a default ModelActivationState."""
    project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
    active_model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
    active_variant_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    other_model_id = UUID("da21744f-990c-4b95-aa2a-70da8d46fdcf")
    available_models = [active_model_id, other_model_id]
    return ModelActivationState(
        project_id=project_id,
        active_model_id=active_model_id,
        active_model_variant_id=active_variant_id,
        available_models=available_models,
        device=DeviceInfo(type=DeviceType.CPU, name="cpu"),
    )


@pytest.fixture
def fxt_active_model_service(fxt_model_activation_state) -> Iterator[ActiveModelService]:
    """Fixture to create an ActiveModelService instance with a temporary data directory."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(ActiveModelService, "_load_state", return_value=fxt_model_activation_state),
    ):
        yield ActiveModelService(data_dir=Path(tmpdir))


class TestActiveModelServiceUnit:
    """Unit tests for ActiveModelService."""

    def test_get_model_file_path(self, fxt_active_model_service):
        """Test retrieval of model file path from variants directory."""
        project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
        model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
        variant_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        extension = "bin"
        expected_path = (
            fxt_active_model_service.projects_dir
            / f"{project_id}/models/{model_id}/variants/{variant_id}/model.{extension}"
        )
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.touch()

        file_path = fxt_active_model_service._get_model_file_path(
            project_id=project_id,
            model_id=model_id,
            variant_id=variant_id,
            extension=extension,
        )

        assert file_path == expected_path

    def test_get_model_file_path_not_found(self, fxt_active_model_service):
        """Test error when model file is not found."""
        project_id = UUID("82d20877-4dd6-4df3-b6bc-418bb300007d")
        model_id = UUID("d4992996-4d87-422b-aaf3-7427267a50df")
        model_variant_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        extension = "bin"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            fxt_active_model_service._get_model_file_path(
                project_id=project_id,
                model_id=model_id,
                variant_id=model_variant_id,
                extension=extension,
            )

    def test_get_loaded_inference_model(self, fxt_active_model_service, fxt_model_activation_state):
        """Test loading the active inference model."""

        with (
            patch.object(
                fxt_active_model_service,
                "_get_model_file_path",
                new=lambda project_id, model_id, variant_id, extension: Path(f"model.{extension}"),
            ),
            patch("app.services.inference.model_loader.ModelLoader.load") as mock_load_model,
        ):
            fxt_active_model_service.get_loaded_inference_model(force_reload=True)

            mock_load_model.assert_called_once_with(
                model_id=fxt_model_activation_state.active_model_id,
                variant_id=fxt_model_activation_state.active_model_variant_id,
                model_xml_path=Path("model.xml"),
                device=fxt_model_activation_state.device,
            )

    def test_force_reload_triggers_unload(self, fxt_active_model_service, tmp_path):
        """
        When force_reload=True is called on a service that already holds a loaded model, model unload must be invoked.
        """
        # Build a fake adapter with the OV native attributes we expect to be cleaned up
        fake_adapter = Mock()
        fake_adapter.async_queue = Mock()
        fake_adapter.compiled_model = Mock()

        fake_model = Mock(spec=Model)
        fake_model.inference_adapter = fake_adapter

        # Inject a pre-loaded model into the service as if inference already ran
        fxt_active_model_service._loaded_model = LoadedModelHandle(
            model_id=uuid4(),
            variant_id=uuid4(),
            model=fake_model,
            device=DeviceInfo(type=DeviceType.CPU, name="cpu"),
            loaded_at=datetime.now(),
        )

        with (
            patch("app.services.inference.model_loader.ModelLoader.load") as mock_load_model,
            patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload_model,
            patch.object(fxt_active_model_service, "_get_model_file_path", return_value=tmp_path / "model.xml"),
        ):
            fxt_active_model_service.get_loaded_inference_model(force_reload=True)
            mock_unload_model.assert_called_once()
            mock_load_model.assert_called_once()
