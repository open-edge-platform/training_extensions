# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from uuid import UUID

from model_api.models import Model
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.repositories import ModelRevisionRepository
from app.schemas.model import Model as ModelSchema
from app.schemas.model_activation import ModelActivationState
from app.services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from app.services.mappers.model_revision_mapper import ModelRevisionMapper
from app.services.parent_process_guard import parent_process_only

logger = logging.getLogger(__name__)

MODELAPI_DEVICE = os.getenv("MODELAPI_DEVICE", "AUTO")
MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class LoadedModel:
    id: UUID
    model: Model


class ModelService:
    """Service to register and activate models"""

    def __init__(self, data_dir: Path, mp_model_reload_event: EventClass | None = None) -> None:
        self.models_dir = data_dir / "models"
        self._mp_model_reload_event = mp_model_reload_event
        self._persistence: GenericPersistenceService[Model, ModelRevisionRepository] = GenericPersistenceService(
            ServiceConfig(ModelRevisionRepository, ModelRevisionMapper, ResourceType.MODEL)
        )
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
                active_model_id=UUID(active_model.id) if active_model is not None else None,
                available_models=[UUID(m.id) for m in available_models],
            )

    def _get_model_xml_path(self, model_id: UUID) -> Path:
        return self.models_dir / f"{model_id}.xml"

    def _get_model_bin_path(self, model_id: UUID) -> Path:
        return self.models_dir / f"{model_id}.bin"

    def get_loaded_inference_model(self, force_reload: bool = False) -> LoadedModel | None:
        """
        Get the currently active model for inference.

        Args:
            force_reload: If True, reload the state and the model from disk. This option can be useful
            to bypass the cache after the state has been modified externally.

        Returns: Model for inference or None if no model is active
        """
        if force_reload:
            self._model_activation_state = self._load_state()
            self._loaded_model = None

        if self._model_activation_state.active_model_id is None:
            return None

        active_model_id = self._model_activation_state.active_model_id
        if self._loaded_model is None or self._loaded_model.id != active_model_id:
            logger.info("Loading model with ID '%s'", active_model_id)
            model_path = self._get_model_xml_path(active_model_id)
            self._loaded_model = LoadedModel(
                id=self._model_activation_state.active_model_id,
                model=Model.create_model(
                    model=str(model_path),
                    device=MODELAPI_DEVICE,
                    nstreams=MODELAPI_NSTREAMS,
                ),
            )
        return self._loaded_model

    def get_model_by_id(self, model_id: UUID, db: Session | None = None) -> ModelSchema:
        """Get a model by its ID"""
        model = self._persistence.get_by_id(model_id, db)
        if not model:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return model

    @parent_process_only
    def delete_model_by_id(self, model_id: UUID) -> None:
        """Delete a model by its ID"""
        with get_db_session() as db:
            self._persistence.delete_by_id(model_id, db)

    def list_models(self) -> list[ModelSchema]:
        """Get information about available models"""
        return self._persistence.list_all()
