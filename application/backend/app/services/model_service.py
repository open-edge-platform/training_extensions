# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB
from app.models import ModelRevision, TrainingStatus
from app.models.training_configuration.configuration import TrainingConfiguration
from app.repositories import LabelRepository, ModelRevisionRepository

from .base import BaseSessionManagedService, ResourceInUseError, ResourceNotFoundError, ResourceType
from .parent_process_guard import parent_process_only


@dataclass(frozen=True)
class ModelRevisionMetadata:
    model_id: UUID
    project_id: UUID
    architecture_id: str
    parent_revision_id: UUID | None
    dataset_revision_id: UUID | None
    training_status: TrainingStatus
    training_configuration: TrainingConfiguration | None = None


class ModelService(BaseSessionManagedService):
    """Service to register and activate models"""

    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self._projects_dir = data_dir / "projects"

    def get_model(self, project_id: UUID, model_id: UUID) -> ModelRevision:
        """
        Get a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.

        Returns:
            ModelRevision: The model revision object containing the model's information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return ModelRevision.model_validate(model_rev_db)

    @parent_process_only
    def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete a model.

        Deletes a model revision from the database and deletes the folder from the filesystem
        associated with this model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to delete.
            model_id (UUID): The unique identifier of the model to delete.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ResourceInUseError: If the model cannot be deleted due to integrity constraints
                (e.g., the model is referenced by other entities).
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)

        path = self._projects_dir / str(project_id) / "models" / str(model_id)
        if path.exists():
            shutil.rmtree(path)

        try:
            deleted = model_rev_repo.delete(str(model_id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.MODEL, str(model_id))

    def list_models(self, project_id: UUID) -> list[ModelRevision]:
        """
        Get information about all available model revisions in a project.

        Retrieves a list of all model revisions that belong to the specified project.
        Each model revision is converted to a schema object containing the model's
        metadata and configuration information.

        Args:
            project_id (UUID): The unique identifier of the project whose models to list.

        Returns:
            list[ModelRevision]: A list of model revision objects representing all model
                revisions in the project. Returns an empty list if the project has no models.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        return [ModelRevision.model_validate(model_rev_db) for model_rev_db in model_rev_repo.list_all()]

    def create_revision(self, metadata: ModelRevisionMetadata) -> None:
        """
        Create and persist a new model revision for the given project metadata.

        Reads the project's label definitions, serializes them into a dict format,
        combines them with the provided metadata into a new model revision record,
        and saves it to the database.

        Args:
            metadata (ModelRevisionMetadata): Metadata used to create the new model revision
                including project id, architecture, optional parent revision id,
                dataset revision id, training status and optional training
                configuration.
        """
        project_id = str(metadata.project_id)
        label_repo = LabelRepository(project_id=project_id, db=self.db_session)
        labels_schema_rev = {"labels": [{"name": label.name, "id": label.id} for label in label_repo.list_all()]}
        model_revision_repo = ModelRevisionRepository(project_id=project_id, db=self.db_session)
        model_revision_repo.save(
            ModelRevisionDB(
                id=str(metadata.model_id),
                project_id=str(metadata.project_id),
                architecture=metadata.architecture_id,
                parent_revision=str(metadata.parent_revision_id) if metadata.parent_revision_id else None,
                training_status=metadata.training_status,
                training_configuration=metadata.training_configuration.model_dump()
                if metadata.training_configuration
                else {},
                training_dataset_id=str(metadata.dataset_revision_id),
                label_schema_revision=labels_schema_rev,
            )
        )
