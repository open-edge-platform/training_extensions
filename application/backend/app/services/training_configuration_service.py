# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from functools import lru_cache
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.model_manifest import ModelManifest
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

    @staticmethod
    @lru_cache(maxsize=8)
    def get_default_by_model_architecture(model_architecture_id: str) -> TrainingConfiguration:
        """
        Retrieve the default training configuration for a given model architecture.

        This is the configuration that will be used to train new model revisions of the specified architecture if no
        custom configuration has been set. The algo-level part of the configuration is based on the model manifest.

        Configurable parameters corresponding to features that are not supported by the model architecture (e.g.
        tiling for an architecture whose ``capabilities.tiling`` is False) are stripped from the configuration so
        that they are not exposed to the user.

        Args:
            model_architecture_id: Identifier for the model architecture.

        Returns:
            TrainingConfiguration: The default training configuration object for the specified model architecture.
        """
        model_manifest = ModelManifestService.get_model_manifest_by_id(model_manifest_id=model_architecture_id)

        training_config = TrainingConfiguration(
            task_level_parameters=TaskLevelParameters(),
            algo_level_parameters=model_manifest.hyperparameters,
        )
        TrainingConfigurationService._strip_unsupported_parameters(training_config, model_manifest)
        return training_config

    @staticmethod
    def _strip_unsupported_parameters(
        training_configuration: TrainingConfiguration, model_manifest: ModelManifest
    ) -> None:
        """
        Remove configurable parameters that correspond to capabilities not supported by the model architecture.

        Parameters are removed by setting them to ``None`` on the configuration. Downstream serialization (e.g. the
        ``TrainingConfigurationView``) skips ``None`` values, ensuring the unsupported parameters are not exposed
        through the API.
        """
        if not model_manifest.capabilities.tiling:
            training_configuration.algo_level_parameters.dataset_preparation.augmentation.tiling = None

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
        # Loading the default configuration also validates that the specified model architecture exists
        default_config = self.get_default_by_model_architecture(model_architecture_id=model_architecture_id)

        training_config_repo = TrainingConfigurationRepository(self.db_session)

        # Load the task-level configuration, or initialize one with default values if not found in the database
        task_level_config_db = training_config_repo.get_task_level_configuration(project_id=str(project_id))
        if task_level_config_db is None:
            task_level_parameters = deepcopy(default_config.task_level_parameters)
        else:
            task_level_parameters = TaskLevelParameters.model_validate(task_level_config_db.configuration_data)

        algo_level_config_db = training_config_repo.get_algo_level_configuration(
            project_id=str(project_id), model_architecture_id=model_architecture_id
        )
        if algo_level_config_db is None:
            algo_level_parameters = deepcopy(default_config.algo_level_parameters)
        else:
            algo_level_parameters = AlgoLevelParameters.model_validate(algo_level_config_db.configuration_data)

        training_configuration = TrainingConfiguration(
            task_level_parameters=task_level_parameters,
            algo_level_parameters=algo_level_parameters,
        )
        # Strip unsupported parameters in case they were persisted (e.g. from a previous manifest version)
        model_manifest = ModelManifestService.get_model_manifest_by_id(model_manifest_id=model_architecture_id)
        self._strip_unsupported_parameters(training_configuration, model_manifest)
        return training_configuration

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
