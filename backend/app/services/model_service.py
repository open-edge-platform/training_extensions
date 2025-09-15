# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from threading import Lock
from uuid import UUID

import aiofiles
from fastapi import UploadFile
from model_api.models import Model
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.db.schema import ModelDB
from app.repositories import ModelRepository
from app.schemas.model import Model as ModelSchema
from app.schemas.model import ModelFormat
from app.schemas.model_activation import ModelActivationState
from app.services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from app.services.mappers.model_mapper import ModelMapper
from app.services.parent_process_guard import parent_process_only

logger = logging.getLogger(__name__)

MODELAPI_DEVICE = os.getenv("MODELAPI_DEVICE", "AUTO")
MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


class ModelAlreadyExistsError(Exception):
    """Exception raised when a model with the same name already exists"""


@dataclass
class LoadedModel:
    name: str
    id: UUID
    model: Model


class ModelService:
    """Service to register and activate models"""

    def __init__(self, mp_model_reload_event: EventClass | None = None) -> None:
        self.models_dir = Path("data/models")
        self._mp_model_reload_event = mp_model_reload_event

        self._persistence: GenericPersistenceService[Model, ModelRepository] = GenericPersistenceService(
            ServiceConfig(ModelRepository, ModelMapper, ResourceType.MODEL)
        )
        self._model_activation_state: ModelActivationState = self._load_state()
        self._model_activation_state_lock = Lock()

        self._loaded_model: LoadedModel | None = None

    @staticmethod
    def _load_state() -> ModelActivationState:
        """Load the state from the file if it exists, otherwise initialize an empty state"""
        with get_db_session() as db:
            repo = ModelRepository(db)
            active_model = repo.get_active_model()
            available_models = repo.list_all()
            return ModelActivationState(
                active_model=active_model.name if active_model is not None else None,
                active_model_id=UUID(active_model.id) if active_model is not None else None,
                available_models=[m.name for m in available_models],
            )

    def _get_model_xml_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.xml"

    def _get_model_bin_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.bin"

    async def _save_files(self, model_name: str, model_xml_file: UploadFile, model_bin_file: UploadFile) -> None:
        xml_path = self._get_model_xml_path(model_name)
        bin_path = self._get_model_bin_path(model_name)

        async def save_file(file_reader: UploadFile, path: Path):
            async with aiofiles.open(path, "wb") as f:
                while chunk := await file_reader.read(1024 * 1024):  # 1MB chunks
                    await f.write(chunk)

        await asyncio.gather(
            save_file(model_xml_file, xml_path),
            save_file(model_bin_file, bin_path),
        )

    async def add_model(self, model_name: str, model_xml_file: UploadFile, model_bin_file: UploadFile) -> ModelSchema:
        """
        Store a new model and make it available for inference

        Args:
            model_name: Name of the model
            model_xml_file: XML file describing the model topology
            model_bin_file: BIN file containing the model weights
        """
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # If a model is already registered with the same name, raise an error
        if model_name in self._model_activation_state.available_models:
            raise ModelAlreadyExistsError(f"A model with the name '{model_name}' already exists")

        with self._model_activation_state_lock:
            # Save the files
            await self._save_files(model_name, model_xml_file, model_bin_file)

            # Add the model to the inference state
            self._model_activation_state.available_models.append(model_name)

            # Activate the model if it is the first model to be added
            is_first_model = self._model_activation_state.active_model is None
            if is_first_model:
                self._model_activation_state.active_model = model_name

            # Store the model in db
            model = ModelDB(name=model_name, format=ModelFormat.OPENVINO)
            with get_db_session() as db:
                repo = ModelRepository(db)
                repo.save(model)
                if is_first_model:
                    repo.set_active_model(model_name)
                db.commit()
            if is_first_model and self._mp_model_reload_event:
                self._mp_model_reload_event.set()
            return ModelMapper.to_schema(model)

    def remove_model(self, model_name: str) -> None:
        """
        Remove a previously added model

        Args:
            model_name: Name of the model to remove
        """
        with self._model_activation_state_lock:
            # If the model does not exist, raise an error
            if model_name not in self._model_activation_state.available_models:
                raise ResourceNotFoundError(ResourceType.MODEL, model_name, f"Model '{model_name}' not found")

            # Remove the model from the inference state
            self._model_activation_state.available_models.remove(model_name)

            # If the model is active, deactivate it and activate the next available model
            next_model = None
            if self._model_activation_state.active_model == model_name:
                logger.info(f"Deactivating model '{model_name}'")
                if len(self._model_activation_state.available_models) > 0:
                    next_model = self._model_activation_state.available_models[0]
                    logger.info(f"Activating next available model '{next_model}'")
                    self._model_activation_state.active_model = next_model
                else:
                    logger.info("No more available models to activate")
                    self._model_activation_state.active_model = None

            # Remove the files
            xml_path = self.models_dir / f"{model_name}.xml"
            bin_path = self.models_dir / f"{model_name}.bin"
            xml_path.unlink()
            bin_path.unlink()

            with get_db_session() as db:
                repo = ModelRepository(db)
                repo.remove(model_name)
                if next_model:
                    repo.set_active_model(next_model)
                db.commit()
            if next_model and self._mp_model_reload_event:
                self._mp_model_reload_event.set()

    def get_available_model_names(self) -> list[str]:
        """Get the names of all available models"""
        return self._model_activation_state.available_models

    def get_active_model_name(self) -> str | None:
        """Get the name of the active model"""
        return self._model_activation_state.active_model

    def activate_model(self, model_name: str) -> None:
        """Activate a model for inference"""
        logger.info(f"Activating model '{model_name}'")
        with self._model_activation_state_lock:
            # If there is no model available with the given name, raise an error
            if model_name not in self._model_activation_state.available_models:
                raise ResourceNotFoundError(ResourceType.MODEL, model_name, f"Model '{model_name}' not found")

            # Activate the model
            self._model_activation_state.active_model = model_name

            # Store the state
            with get_db_session() as db:
                repo = ModelRepository(db)
                repo.set_active_model(model_name)
                db.commit()
            if self._mp_model_reload_event:
                self._mp_model_reload_event.set()

    def get_loaded_inference_model(self, force_reload: bool = False) -> LoadedModel | None:
        """
        Get the currently active model for inference.

        Args:
            force_reload: If True, reload the state and the model from disk. This option can be useful
            to bypass the cache after the state has been modified externally.

        Returns: Model for inference or None if no model is active
        """
        if force_reload:
            with self._model_activation_state_lock:
                self._model_activation_state = self._load_state()
                self._loaded_model = None

        if self._model_activation_state.active_model is None:
            return None

        if self._model_activation_state.active_model_id is None:
            return None

        if self._loaded_model is None or self._loaded_model.name != self._model_activation_state.active_model:
            logger.info(f"Loading model '{self._model_activation_state.active_model}'")
            model_path = self._get_model_xml_path(self._model_activation_state.active_model)
            self._loaded_model = LoadedModel(
                name=self._model_activation_state.active_model,
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

    @parent_process_only
    def update_model(self, model_id: UUID, model_metadata: dict) -> ModelSchema:
        """Update the metadata of an existing model"""
        with get_db_session() as db:
            model = self.get_model_by_id(model_id, db)
            return self._persistence.update(model, model_metadata, db)
