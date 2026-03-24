# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from loguru import logger
from model_api.adapters import OpenvinoAdapter, create_core
from model_api.models import Model

from app.db.engine import get_db_session
from app.models.model_activation import ModelActivationState
from app.models.model_revision import ModelFormat, ModelPrecision
from app.repositories import ModelRevisionRepository, ModelVariantRepository
from app.repositories.active_model_repo import ActiveModelRepo
from app.utils.ir_format import needs_float32_input

MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


class _FP32OpenvinoAdapter(OpenvinoAdapter):
    """OpenvinoAdapter that forces float32 input tensors.

    Used when the IR embeds mean/std in the 0-1 scale (new OTX format).
    Overrides ``embed_preprocessing`` so ModelAPI sets the input tensor to f32.
    """

    def embed_preprocessing(self, *args, **kwargs) -> None:
        kwargs["dtype"] = float
        super().embed_preprocessing(*args, **kwargs)


@dataclass(frozen=True)
class LoadedModel:
    model_revision_id: UUID
    model_variant_id: UUID
    model: Model
    device: str
    float32_input: bool  # True → InferenceWorker must scale images to [0,1] float32


@dataclass(frozen=True)
class DeviceType:
    """Value object representing a device type with optional index."""

    name: str
    index: int | None = None

    def __str__(self) -> str:
        """Convert to OpenVINO device string format (e.g., 'GPU.1')."""
        return f"{self.name.upper()}.{self.index}" if self.index is not None else self.name.upper()

    @classmethod
    def from_raw(cls, raw_device_name: str) -> "DeviceType":
        """
        Parse raw device name into DeviceType.
        Examples:
            "cpu" -> DeviceType(name="CPU", index=None)
            "xpu" -> DeviceType(name="GPU", index=None)
            "xpu-1" -> DeviceType(name="GPU", index=1)
        """
        if raw_device_name.lower() == "cpu":
            return cls(name="CPU")

        if raw_device_name.lower().startswith("xpu"):
            parts = raw_device_name.split("-")
            if len(parts) == 1:
                return DeviceType(name="GPU")
            if len(parts) == 2 and parts[1].isdigit():
                return DeviceType(name="GPU", index=int(parts[1]))
        raise ValueError(f"Unsupported device name: {raw_device_name}")


class ActiveModelService:
    """
    Service to fetch the currently active model for inference.

    Used exclusively by the InferenceWorker process.
    """

    def __init__(self, data_dir: Path) -> None:
        self.projects_dir = data_dir / "projects"
        self._model_activation_state: ModelActivationState = self._load_state()
        self._loaded_model: LoadedModel | None = None

    @staticmethod
    def _load_state() -> ModelActivationState:
        """Load the state from the file if it exists, otherwise initialize an empty state"""
        with get_db_session() as db:
            active_model_repo = ActiveModelRepo(db=db)
            active_model = active_model_repo.get_active_revision()
            if active_model is None:
                return ModelActivationState(
                    project_id=None,
                    active_model_id=None,
                    active_model_variant_id=None,
                    available_models=[],
                    device="",
                )
            model_rev_repo = ModelRevisionRepository(project_id=str(active_model.project_id), db=db)
            available_models = model_rev_repo.list_all()
            model_variants_repo = ModelVariantRepository(db=db)
            model_variants = model_variants_repo.list_by_model_revision(str(active_model.id))
            active_variant_id = next(
                v.id for v in model_variants if v.format == ModelFormat.OPENVINO and v.precision == ModelPrecision.FP16
            )
            pipeline_device = active_model_repo.get_active_pipeline_device()
            if pipeline_device is None:
                raise RuntimeError("Active pipeline must have a device configured")
            ov_device = DeviceType.from_raw(pipeline_device)
            return ModelActivationState(
                project_id=UUID(active_model.project_id),
                active_model_id=UUID(active_model.id),
                active_model_variant_id=UUID(active_variant_id),
                available_models=[UUID(m.id) for m in available_models],
                device=str(ov_device),
            )

    def _get_model_file_path(self, project_id: UUID, model_id: UUID, variant_id: UUID, extension: str = "xml") -> Path:
        file_path = self.projects_dir / f"{project_id}/models/{model_id}/variants/{variant_id}/model.{extension}"
        if file_path.is_file():
            return file_path
        raise FileNotFoundError(f"Model file not found: {file_path}")

    def get_loaded_inference_model(self, force_reload: bool = False) -> LoadedModel | None:
        """
        Get the currently active model for inference.

        Args:
            force_reload: If True, reload the state and the model from disk. This option can be useful
            to bypass the cache after the state has been modified externally.

        Returns: Model for inference or None if no model is active, or if the model can't be loaded.
        """
        if force_reload:
            self._model_activation_state = self._load_state()
            self._loaded_model = None

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
            or self._loaded_model.model_revision_id != active_model_id
            or self._loaded_model.model_variant_id != active_variant_id
            or self._loaded_model.device != device
        )
        if needs_reload:
            logger.info(
                "Loading model with ID '{}', variant '{}', on device '{}'", active_model_id, active_variant_id, device
            )
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
                use_float32 = needs_float32_input(model_xml_path)
                ie = create_core()
                adapter_cls = _FP32OpenvinoAdapter if use_float32 else OpenvinoAdapter
                logger.info(
                    "IR format detected: {} (float32_input={})",
                    model_xml_path.name,
                    use_float32,
                )
                adapter = adapter_cls(
                    ie,
                    str(model_xml_path),
                    device=device,
                    max_num_requests=int(MODELAPI_NSTREAMS),
                )
                mapi_model = Model.create_model(adapter)
            except FileNotFoundError:
                logger.exception("Failed to load model with ID '{}'", active_model_id)
                return None

            self._loaded_model = LoadedModel(
                model_revision_id=self._model_activation_state.active_model_id,
                model=mapi_model,
                model_variant_id=active_variant_id,
                device=device,
                float32_input=use_float32,
            )
        return self._loaded_model
