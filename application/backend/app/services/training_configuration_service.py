# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.training_configuration import AlgoLevelParameters, TaskLevelParameters, TrainingConfiguration
from app.repositories import ModelRevisionRepository
from app.repositories.training_configuration_repo import TrainingConfigurationRepository
from app.services import BaseSessionManagedService, ModelManifestService, ResourceNotFoundError, ResourceType


class TrainingConfigurationService(BaseSessionManagedService):
    def __init__(self, db_session: Session | None = None) -> None:
        super().__init__(db_session)

    def get_by_model_revision(self, project_id: UUID, model_revision_id: UUID) -> TrainingConfiguration:
        """
        Retrieve the configuration used to train a specific model revision.

        Args:
            project_id: Identifier for the project.
            model_revision_id: Identifier for the model revision.

        Returns:
            TrainingConfiguration: The training configuration object associated with the specified model revision.

        Raises:
            ResourceNotFoundError: If the specified model revision does not exist.
        """
        model = ModelRevisionRepository(str(project_id), self.db_session).get_by_id(str(model_revision_id))
        if not model:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_revision_id))
        return TrainingConfiguration.model_validate(model.training_configuration)

    def get_by_model_architecture(self, project_id: UUID, model_architecture_id: str) -> TrainingConfiguration:
        """
        Retrieve the current training configuration for a given model architecture within a project.

        This is the configuration that will be used to train new model revisions of the specified architecture.
        If no specific configuration has been set for the model architecture, a default configuration will be returned
        based on the model manifest.

        Args:
            project_id: Identifier for the project.
            model_architecture_id: Identifier for the model architecture.

        Returns:
            TrainingConfiguration: The training configuration object associated with the specified model architecture.

        Raises:
            ManifestNotFoundException: If the specified model architecture does not exist.
        """
        # Validate that the specified model architecture exists by loading its manifest
        model_manifest = ModelManifestService.get_model_manifest_by_id(model_manifest_id=model_architecture_id)

        training_config_repo = TrainingConfigurationRepository(self.db_session)

        # Load the task-level configuration, or initialize one with default values if not found in the database
        task_level_config_db = training_config_repo.get_task_level_configuration(project_id=str(project_id))
        if task_level_config_db is None:
            task_level_parameters = TaskLevelParameters()
        else:
            task_level_parameters = TaskLevelParameters.model_validate(task_level_config_db.configuration_data)

        algo_level_config_db = training_config_repo.get_algo_level_configuration(
            project_id=str(project_id), model_architecture_id=model_architecture_id
        )
        if algo_level_config_db is None:
            algo_level_parameters = model_manifest.hyperparameters
        else:
            algo_level_parameters = AlgoLevelParameters.model_validate(algo_level_config_db.configuration_data)

        return TrainingConfiguration(
            task_level_parameters=task_level_parameters,
            algo_level_parameters=algo_level_parameters,
        )

    def update(
        self, project_id: UUID, model_architecture_id: str, training_configuration: TrainingConfiguration
    ) -> None:
        """
        Update the training configuration.

        This method can handle both updates to task-level parameters (which apply to all model architectures within
        the project) and algo-level parameters (which are specific to a model architecture). In practice, the method
        persists all the parameters values set in the provided TrainingConfiguration object, automatically determining
        which parameters are task-level and which are algo-level.

        Args:
            project_id: Identifier for the project.
            model_architecture_id: Identifier for the model architecture to which the algo-level parameters apply.
            training_configuration: The TrainingConfiguration object containing the updated configuration values.

        Raises:
            ManifestNotFoundException: If the specified model architecture does not exist.
        """
        # Validate that the specified model architecture exists by loading its manifest
        ModelManifestService.get_model_manifest_by_id(model_manifest_id=model_architecture_id)

        training_config_repo = TrainingConfigurationRepository(self.db_session)

        # Update the task-level configuration
        training_config_repo.create_or_update(
            project_id=str(project_id),
            model_architecture_id=None,
            configuration_data=training_configuration.task_level_parameters.model_dump(),
        )
        # Update the algo-level configuration
        training_config_repo.create_or_update(
            project_id=str(project_id),
            model_architecture_id=model_architecture_id,
            configuration_data=training_configuration.algo_level_parameters.model_dump(),
        )
