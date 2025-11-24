# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from uuid import UUID

from loguru import logger
from model_api.models import Model

from app.db.engine import get_db_session
from app.repositories import ModelRevisionRepository
from app.repositories.active_model_repo import ActiveModelRepo
from app.schemas.model_activation import ModelActivationState

MODELAPI_DEVICE = os.getenv("MODELAPI_DEVICE", "AUTO")
MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class LoadedModel:
    id: UUID
    model: Model


class ActiveModelService:
    """
    Service to fetch the currently active model for inference.

    Used exclusively by the InferenceWorker process.
    """

    def __init__(self, data_dir: Path, mp_model_reload_event: EventClass) -> None:
        self.projects_dir = data_dir / "projects"
        self._mp_model_reload_event = mp_model_reload_event
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
                    available_models=[],
                )
            model_rev_repo = ModelRevisionRepository(project_id=str(active_model.project_id), db=db)
            available_models = model_rev_repo.list_all()
            return ModelActivationState(
                project_id=UUID(active_model.project_id),
                active_model_id=UUID(active_model.id),
                available_models=[UUID(m.id) for m in available_models],
            )

    def _get_model_file_path(self, project_id: UUID, model_id: UUID, extension: str = "xml") -> Path:
        file_path = self.projects_dir / f"{project_id}/models/{model_id}/model.{extension}"
        if not file_path.is_file():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        return file_path

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

        if self._model_activation_state.active_model_id is None or self._model_activation_state.project_id is None:
            return None

        project_id = self._model_activation_state.project_id
        active_model_id = self._model_activation_state.active_model_id
        if self._loaded_model is None or self._loaded_model.id != active_model_id:
            logger.info("Loading model with ID '{}'", active_model_id)
            try:
                # Ensure all necessary model files exist before loading the model
                model_xml_path = self._get_model_file_path(project_id, active_model_id, "xml")
                _ = self._get_model_file_path(project_id, active_model_id, "bin")
                mapi_model = Model.create_model(
                    model=str(model_xml_path),
                    device=MODELAPI_DEVICE,
                    nstreams=MODELAPI_NSTREAMS,
                )
            except FileNotFoundError:
                logger.exception("Failed to load model with ID '{}'", active_model_id)
                return None

            self._loaded_model = LoadedModel(
                id=self._model_activation_state.active_model_id,
                model=mapi_model,
            )
        return self._loaded_model
