# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from loguru import logger

from app.db.engine import get_db_session
from app.models.model_activation import ModelActivationState
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.repositories import ModelRevisionRepository, ModelVariantRepository
from app.repositories.active_model_repo import ActiveModelRepo
from app.services.inference.model_loader import LoadedModelHandle, ModelLoader

from .system_service import SystemService


class ActiveModelService:
    """
    Service to fetch the currently active model for inference.

    Used exclusively by the InferenceWorker process.
    """

    def __init__(self, data_dir: Path) -> None:
        self.projects_dir = data_dir / "projects"
        self._model_activation_state: ModelActivationState = self._load_state()
        self._loaded_model: LoadedModelHandle | None = None

    @property
    def active_project_id(self) -> UUID | None:
        """Project ID of the currently active model, or None if no model is active."""
        return self._model_activation_state.project_id

    @staticmethod
    def _load_state() -> ModelActivationState:
        """Load the state from the DB if it exists, otherwise initialize an empty state"""
        with get_db_session() as db:
            active_model_repo = ActiveModelRepo(db=db)
            active_model = active_model_repo.get_active_revision()
            if active_model is None:
                return ModelActivationState(
                    project_id=None,
                    active_model_id=None,
                    active_model_variant_id=None,
                    available_models=[],
                    device=DeviceInfo(type=DeviceType.CPU, name="cpu"),
                )
            model_rev_repo = ModelRevisionRepository(project_id=str(active_model.project_id), db=db)
            available_models = model_rev_repo.list_all(training_status=TrainingStatus.SUCCESSFUL)
            # Use the variant configured in the pipeline, fall back to FP16 OpenVINO
            active_variant_id = active_model_repo.get_active_model_variant_id()
            if active_variant_id is None:
                model_variants_repo = ModelVariantRepository(db=db)
                model_variants = model_variants_repo.list_by_model_revision(str(active_model.id))
                active_variant_id = next(
                    v.id
                    for v in model_variants
                    if v.format == ModelFormat.OPENVINO and v.precision == ModelPrecision.FP16
                )
                logger.warning("No active model variant ID found, loaded fallback model %s", active_variant_id)
            pipeline_device = active_model_repo.get_active_pipeline_device()
            if pipeline_device is None:
                raise RuntimeError("Active pipeline must have a device configured")
            geti_device = SystemService().get_device_info(pipeline_device)
            return ModelActivationState(
                project_id=UUID(active_model.project_id),
                active_model_id=UUID(active_model.id),
                active_model_variant_id=UUID(active_variant_id),
                available_models=[UUID(m.id) for m in available_models],
                device=geti_device,
            )

    def _get_model_file_path(self, project_id: UUID, model_id: UUID, variant_id: UUID, extension: str = "xml") -> Path:
        file_path = self.projects_dir / f"{project_id}/models/{model_id}/variants/{variant_id}/model.{extension}"
        if file_path.is_file():
            return file_path
        raise FileNotFoundError(f"Model file not found: {file_path}")

    def get_loaded_inference_model(self, force_reload: bool = False) -> LoadedModelHandle | None:
        """
        Get the currently active model for inference.

        Args:
            force_reload: If True, reload the state and the model from disk. This option can be useful
            to bypass the cache after the state has been modified externally.

        Returns: Model for inference or None if no model is active, or if the model can't be loaded.
        """
        if force_reload:
            self._unload_model()
            self._model_activation_state = self._load_state()

        if (
            self._model_activation_state.active_model_id is None
            or self._model_activation_state.active_model_variant_id is None
            or self._model_activation_state.project_id is None
        ):
            return None

        project_id = self._model_activation_state.project_id
        active_model_id = self._model_activation_state.active_model_id
        active_variant_id = self._model_activation_state.active_model_variant_id
        device = self._model_activation_state.device
        needs_reload = (
            self._loaded_model is None
            or self._loaded_model.model_id != active_model_id
            or self._loaded_model.variant_id != active_variant_id
            or self._loaded_model.device != device
        )
        if needs_reload:
            logger.info(
                "Loading model with ID '{}', variant '{}', on device '{}'", active_model_id, active_variant_id, device
            )
            self._unload_model()
            try:
                # Ensure all necessary model files exist before loading the model
                model_xml_path = self._get_model_file_path(
                    project_id=project_id,
                    model_id=active_model_id,
                    variant_id=active_variant_id,
                    extension="xml",
                )
                _ = self._get_model_file_path(
                    project_id=project_id,
                    model_id=active_model_id,
                    variant_id=active_variant_id,
                    extension="bin",
                )
                self._loaded_model = ModelLoader.load(
                    model_id=active_model_id, variant_id=active_variant_id, model_xml_path=model_xml_path, device=device
                )
            except FileNotFoundError:
                logger.exception("Failed to load model with ID '{}'", active_model_id)
                return None

        return self._loaded_model

    def _unload_model(self) -> None:
        """Release the currently loaded model and free its resources."""
        if self._loaded_model is not None:
            logger.debug("Unloading model '{}'", self._loaded_model.model_id)
            ModelLoader.unload(self._loaded_model)
            self._loaded_model = None
