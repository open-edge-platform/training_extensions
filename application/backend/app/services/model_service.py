# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.repositories import ModelRevisionRepository, ProjectRepository
from app.schemas.model import Model as ModelSchema

from .base import ResourceInUseError, ResourceNotFoundError, ResourceType
from .mappers.model_revision_mapper import ModelRevisionMapper
from .parent_process_guard import parent_process_only

logger = logging.getLogger(__name__)


class ModelService:
    """Service to register and activate models"""

    def __init__(self, db_session: Session) -> None:
        self._db_session = db_session

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
        project_repo = ProjectRepository(self._db_session)
        # Prefer using a JOIN here since the list of model revisions per project is not large,
        # and it allows us to check for project existence and fetch the model in a single query.
        project = project_repo.get_by_id(str(project_id))
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        model = next((ModelRevisionMapper.to_schema(m) for m in project.model_revisions if m.id == str(model_id)), None)
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
        project_repo = ProjectRepository(self._db_session)
        if not project_repo.exists(str(project_id)):
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        model_rev_repo = ModelRevisionRepository(self._db_session)
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
        project_repo = ProjectRepository(self._db_session)
        project = project_repo.get_by_id(str(project_id))
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        return [ModelRevisionMapper.to_schema(model_rev_db) for model_rev_db in project.model_revisions]
