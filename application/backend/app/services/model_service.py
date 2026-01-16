# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB
from app.models import ModelRevision, TrainingStatus
from app.models.model_revision import ModelFormat, ModelPrecision
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

    def get_model_variants(self, project_id: UUID, model_id: UUID) -> list[dict]:
        """
        Get all variants and their information of a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve variants for.

        Returns:
            list[dict]: A list of the models variants.
        """
        model_variants = []
        for format in ModelFormat:
            exists, paths = self.get_model_binary_files(project_id=project_id, model_id=model_id, format=format)
            if exists:
                model_size = sum(path.stat().st_size for path in paths)
                model_info = {
                    "format": format.value,
                    "precision": ModelPrecision.FP16 if format != ModelFormat.PYTORCH else ModelPrecision.FP32,
                    "weights_size": model_size,
                }
                model_variants.append(model_info)

        return model_variants

    def rename_model(self, project_id: UUID, model_id: UUID, model_metadata: dict[str, str]) -> ModelRevision:
        """
        Rename a model revision.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.
            model_metadata: Dict containing updated model revision name

        Returns:
            ModelRevision: The model revision object containing the model's updated information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        new_name = model_metadata.get("name")
        if new_name is not None:
            model_rev_db.name = new_name
            model_rev_repo.update(model_rev_db)
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

    @parent_process_only
    def delete_model_files(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete only the model files from disk, keeping the model revision record in the database and setting its
        files_deleted flag to True.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
        """
        for model_format in ModelFormat:
            exists, _ = self.get_model_binary_files(project_id=project_id, model_id=model_id, format=model_format)

            # Delete model files from disk, if at least 1 model file is present
            if exists:
                path = self._projects_dir / str(project_id) / "models" / str(model_id)
                shutil.rmtree(path)
                logger.info("Deleted model files at '{}'", path)
                break

        # Mark as deleted in the database
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = cast(ModelRevisionDB, model_rev_repo.get_by_id(str(model_id)))
        model_rev_db.files_deleted = True
        model_rev_repo.update(model_rev_db)

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
                name=f"{metadata.architecture_id} ({str(metadata.model_id).split('-')[0]})",
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

    def get_model_binary_files(
        self, project_id: UUID, model_id: UUID, format: ModelFormat
    ) -> tuple[bool, tuple[Path, ...]]:
        """
        Get the paths to the model binary files.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            format (ModelFormat): The format of the model files to retrieve.

        Returns:
            tuple[bool, tuple[Path, ...]]: A tuple where the first element indicates if the files exist,
                and the second element is a tuple of Paths to the model files.

        Raises:
            ResourceNotFoundError: If the model has been marked as deleted.
            FileNotFoundError: If the model directory does not exist.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            return False, ()

        model_dir = self._projects_dir / str(project_id) / "models" / str(model_id)
        xml_file = model_dir / "model.xml"
        bin_file = model_dir / "model.bin"
        onnx_file = model_dir / "model.onnx"
        ckpt_file = model_dir / "model.ckpt"

        if format == ModelFormat.OPENVINO and xml_file.exists() and bin_file.exists():
            return True, (xml_file, bin_file)
        if format == ModelFormat.ONNX and onnx_file.exists():
            return True, (onnx_file,)
        if format == ModelFormat.PYTORCH and ckpt_file.exists():
            return True, (ckpt_file,)

        return False, ()
