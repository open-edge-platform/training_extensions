# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import ModelRevisionRepository, ProjectRepository
from app.repositories.training_configuration_repo import TrainingConfigurationRepository
from app.schemas import TrainingConfiguration
from app.services import ResourceNotFoundError, ResourceType
from app.supported_models import SupportedModels
from app.supported_models.default_models import DefaultModels


class TrainingConfigurationService:
    def __init__(self, db_session: Session) -> None:
        self._db_session = db_session
        self._training_config_repo = TrainingConfigurationRepository(db_session)

    def get_training_configuration(
        self,
        project_id: UUID,
        model_architecture_id: str | None = None,
        model_revision_id: UUID | None = None,
    ) -> TrainingConfiguration:
        """
        Retrieves training configuration.

        If model_revision_id is provided, the configuration is loaded from the model entity.
        If model_architecture_id is provided, tries to load from database first, then from manifest.
        Otherwise, loads default configuration based on project's task type.

        Args:
            project_id (UUID): Identifier for the project.
            model_architecture_id (str | None): Optional ID of the model architecture to retrieve configurations.
            model_revision_id (UUID | None): Optional ID of the model revision to retrieve specific configurations.

        Returns:
            TrainingConfiguration: The training configuration object.
        """
        if model_architecture_id and model_revision_id:
            raise ValueError("Only one of model_architecture_id or model_revision_id should be provided.")

        if model_revision_id:
            model = ModelRevisionRepository(self._db_session).get_by_id(str(model_revision_id))
            if not model:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_revision_id))
            return TrainingConfiguration.from_model(model_config=model.training_configuration)

        if model_architecture_id:
            stored_config = TrainingConfigurationRepository(self._db_session).get_by_project_and_model_architecture(
                project_id=str(project_id),
                model_architecture_id=model_architecture_id,
            )
            if stored_config:
                return TrainingConfiguration.model_validate(stored_config.configuration_data)

            model_manifest = SupportedModels.get_model_manifest_by_id(model_architecture_id)
            return TrainingConfiguration.from_hyperparameters(hyperparameters=model_manifest.hyperparameters)

        # Load default configuration based on the project's task type
        project = ProjectRepository(self._db_session).get_by_id(str(project_id))
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        default_model_id = DefaultModels.get_default_model(task_type=project.task_type)
        if not default_model_id:
            raise ValueError(f"No default model found for task type {project.task_type}")
        default_model_manifest = SupportedModels.get_model_manifest_by_id(model_manifest_id=default_model_id)

        return TrainingConfiguration.from_hyperparameters(hyperparameters=default_model_manifest.hyperparameters)

    def update_training_configuration(
        self,
        project_id: UUID,
        training_config_update: dict,
        model_architecture_id: str | None = None,
    ) -> TrainingConfiguration:
        """
        Updates training configuration with provided changes.

        Args:
            project_id (UUID): Identifier for the project.
            training_config_update (dict): Configuration updates to apply.
            model_architecture_id (str | None): Optional ID of the model architecture.

        Returns:
            TrainingConfiguration: The updated training configuration object.
        """
        project = ProjectRepository(self._db_session).get_by_id(str(project_id))
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        current_config = self.get_training_configuration(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
        )

        validated_update_config = TrainingConfiguration.from_model(training_config_update)
        updated_config = current_config.model_copy(
            update={
                "dataset_preparation": validated_update_config.dataset_preparation,
                "training": validated_update_config.training,
                "evaluation": validated_update_config.evaluation,
            },
            deep=True,
        )

        self._training_config_repo.create_or_update(
            project_id=str(project_id),
            model_architecture_id=model_architecture_id,
            configuration_data=updated_config.model_dump(),
        )

        return updated_config
