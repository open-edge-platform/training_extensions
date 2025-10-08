# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from uuid import UUID

from model_api.models import Model

from app.db.engine import get_db_session
from app.repositories import ModelRevisionRepository
from app.schemas.model_activation import ModelActivationState

logger = logging.getLogger(__name__)

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
            repo = ModelRevisionRepository(db)
            active_model = repo.get_active_revision()
            available_models = repo.list_all()
            return ModelActivationState(
                project_id=UUID(active_model.project_id) if active_model is not None else None,
                active_model_id=UUID(active_model.id) if active_model is not None else None,
                available_models=[UUID(m.id) for m in available_models],
            )

    def _get_model_xml_path(self, project_id: UUID, model_id: UUID) -> Path:
        return self.projects_dir / f"{project_id}/models/{model_id}/model.xml"

    def _get_model_bin_path(self, project_id: UUID, model_id: UUID) -> Path:
        return self.projects_dir / f"{project_id}/models/{model_id}/model.bin"

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
            logger.info("Loading model with ID '%s'", active_model_id)
            model_xml_path = self._get_model_xml_path(project_id, active_model_id)
            model_bin_path = self._get_model_bin_path(project_id, active_model_id)
            if not os.path.isfile(model_xml_path):
                logger.error("Model XML file not found at path: %s", model_xml_path)
                return None
            if not os.path.isfile(model_bin_path):
                logger.error("Model BIN file not found at path: %s", model_bin_path)
                return None
            try:
                mapi_model = Model.create_model(
                    model=str(model_xml_path),
                    device=MODELAPI_DEVICE,
                    nstreams=MODELAPI_NSTREAMS,
                )
            except Exception:
                logger.exception("Failed to create Model API model from '%s'", model_xml_path)
                return None
            self._loaded_model = LoadedModel(
                id=self._model_activation_state.active_model_id,
                model=mapi_model,
            )
        return self._loaded_model
