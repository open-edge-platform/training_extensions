# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from model_api.models import Model

from app.models.model_activation import ModelActivationState
from app.services import ActiveModelService
from app.services.active_model_service import DeviceType, LoadedModel


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
        device="CPU",
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

    @pytest.mark.parametrize(
        "raw_device_name, expected_ov_device_name",
        [
            ("cpu", "CPU"),
            ("xpu", "GPU"),
            ("xpu-0", "GPU.0"),
            ("xpu-1", "GPU.1"),
        ],
    )
    def test_get_ov_device_name(self, raw_device_name, expected_ov_device_name) -> None:
        """Test conversion of raw device names to OpenVINO device names."""
        ov_device_name = DeviceType.from_raw(raw_device_name)
        assert str(ov_device_name) == expected_ov_device_name

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

    def test_get_loaded_inference_model(self, fxt_active_model_service, fxt_model_activation_state, monkeypatch):
        """Test loading the active inference model (uint8-scale IR, float32_input=False)."""
        dummy_model = Mock(spec=Model)

        with (
            patch.object(Model, "create_model", return_value=dummy_model),
            patch.object(
                fxt_active_model_service,
                "_get_model_file_path",
                new=lambda project_id, model_id, variant_id, extension: Path(f"model.{extension}"),
            ),
            patch("app.services.active_model_service.needs_float32_input", return_value=False),
            patch("app.services.active_model_service.create_core"),
            patch("app.services.active_model_service.OpenvinoAdapter"),
        ):
            loaded = fxt_active_model_service.get_loaded_inference_model(force_reload=True)
            assert isinstance(loaded, LoadedModel)
            assert loaded.model_revision_id == fxt_model_activation_state.active_model_id
            assert loaded.model is dummy_model
            assert loaded.device == fxt_model_activation_state.device
            assert loaded.float32_input is False

    def test_get_loaded_inference_model_fp32(self, fxt_active_model_service, fxt_model_activation_state):
        """IR with 0-1 normalisation scale → float32_input=True and _FP32OpenvinoAdapter is used."""
        dummy_model = Mock(spec=Model)

        with (
            patch.object(Model, "create_model", return_value=dummy_model),
            patch.object(
                fxt_active_model_service,
                "_get_model_file_path",
                new=lambda project_id, model_id, variant_id, extension: Path(f"model.{extension}"),
            ),
            patch("app.services.active_model_service.needs_float32_input", return_value=True),
            patch("app.services.active_model_service.create_core"),
            patch("app.services.active_model_service.FP32OpenvinoAdapter") as mock_fp32_adapter,
        ):
            loaded = fxt_active_model_service.get_loaded_inference_model(force_reload=True)

        assert isinstance(loaded, LoadedModel)
        assert loaded.float32_input is True
        mock_fp32_adapter.assert_called_once()
