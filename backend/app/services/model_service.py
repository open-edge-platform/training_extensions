# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from uuid import UUID

from model_api.models import Model
from sqlalchemy.exc import IntegrityError

from app.db import get_db_session
from app.repositories import ModelRevisionRepository, ProjectRepository
from app.schemas.model import Model as ModelSchema
from app.schemas.model_activation import ModelActivationState
from app.services.base import ResourceInUseError, ResourceNotFoundError, ResourceType
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
        self._model_activation_state: ModelActivationState = self._load_state()
        self._loaded_model: LoadedModel | None = None
        self._mapper = ModelRevisionMapper()

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

    def get_model_by_id(self, project_id: UUID, model_id: UUID) -> ModelSchema:
        """
        Get a model by its ID within a specific project.

        Retrieves a model revision from the specified project by matching the model ID.
        The method first validates that the project exists, then searches through the
        project's model revisions to find the one with the matching ID.

        Args:
            project_id (UUID): The unique identifier of the project containing the model.
            model_id (UUID): The unique identifier of the model to retrieve.

        Returns:
            ModelSchema: The model schema object containing the model's information.

        Raises:
            ResourceNotFoundError: If the project with the given project_id does not exist,
                or if no model with the given model_id is found within the project.
        """
        with get_db_session() as db:
            project_repo = ProjectRepository(db)
            # Prefer using a JOIN here since the list of model revisions per project is not large,
            # and it allows us to check for project existence and fetch the model in a single query.
            project = project_repo.get_by_id(str(project_id))
            if not project:
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            model = next((self._mapper.to_schema(m) for m in project.model_revisions if m.id == str(model_id)), None)
            if not model:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
            return model

    @parent_process_only
    def delete_model_by_id(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete a model by its ID from a specific project.

        Permanently removes a model revision from the specified project. The method
        first validates that the project exists, then attempts to delete the model
        from the database. This operation is restricted to the parent process only.

        Args:
            project_id (UUID): The unique identifier of the project containing the model.
            model_id (UUID): The unique identifier of the model to delete.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If the project with the given project_id does not exist,
                or if no model with the given model_id is found.
            ResourceInUseError: If the model cannot be deleted due to integrity constraints
                (e.g., the model is referenced by other entities).
        """
        with get_db_session() as db:
            project_repo = ProjectRepository(db)
            if not project_repo.exists(str(project_id)):
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            model_rev_repo = ModelRevisionRepository(db)
            try:
                # TODO: delete model artifacts from filesystem when implemented
                deleted = model_rev_repo.delete(str(model_id))
                if not deleted:
                    raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
            except IntegrityError:
                raise ResourceInUseError(ResourceType.MODEL, str(model_id))

    def list_models(self, project_id: UUID) -> list[ModelSchema]:
        """
        Get information about all available model revisions in a project.

        Retrieves a list of all model revisions that belong to the specified project.
        Each model revision is converted to a schema object containing the model's
        metadata and configuration information.

        Args:
            project_id (UUID): The unique identifier of the project whose models to list.

        Returns:
            list[ModelSchema]: A list of model schema objects representing all model
                revisions in the project. Returns an empty list if the project has no models.

        Raises:
            ResourceNotFoundError: If the project with the given project_id does not exist.
        """
        with get_db_session() as db:
            project_repo = ProjectRepository(db)
            project = project_repo.get_by_id(str(project_id))
            if not project:
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            return [self._mapper.to_schema(model_rev_db) for model_rev_db in project.model_revisions]
