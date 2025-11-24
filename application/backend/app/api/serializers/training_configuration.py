# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections import defaultdict
from typing import Any

from app.api.serializers.configurable_parameters import ConfigurableParametersConverter
from app.models.training_configuration.configuration import (
    GlobalDatasetPreparationParameters,
    GlobalParameters,
    PartialTrainingConfiguration,
    TrainingConfiguration,
)
from app.models.training_configuration.hyperparameters import (
    DatasetPreparationParameters,
    EvaluationParameters,
    Hyperparameters,
    TrainingHyperParameters,
)
from app.supported_models import SupportedModels

DATASET_PREPARATION = "dataset_preparation"
TRAINING = "training"
EVALUATION = "evaluation"


class TrainingConfigurationConverter(ConfigurableParametersConverter):
    """
    Converters between objects and their corresponding REST views
    """

    @classmethod
    def _dataset_preparation_to_rest(
        cls,
        global_parameters: GlobalParameters | None,
        hyperparameters: Hyperparameters | None,
        default_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Convert dataset preparation parameters to REST representation.

        Combines global parameters and hyperparameters for dataset preparation
        into a unified REST view.

        Args:
            global_parameters (GlobalParameters | None): Global configuration parameters.
            hyperparameters (Hyperparameters | None): Training hyperparameters.
            default_config (dict[str, Any] | None): Optional default configuration for reference.

        Returns:
            dict[str, Any]: Combined REST representation of dataset preparation parameters.
        """
        # Return a combined view of global and hyperparameters for dataset preparation
        default_config = default_config or {}
        global_parameters_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=global_parameters.dataset_preparation,
                default_config=default_config.get("dataset_preparation"),
            )
            if global_parameters and global_parameters.dataset_preparation
            else {}
        )
        hyperparameters_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=hyperparameters.dataset_preparation,
                default_config=default_config.get("dataset_preparation"),
            )
            if hyperparameters and hyperparameters.dataset_preparation
            else {}
        )
        if not isinstance(global_parameters_rest, dict) or not isinstance(hyperparameters_rest, dict):
            raise ValueError("Expected dictionary for global and hyperparameters REST views")
        return global_parameters_rest | hyperparameters_rest

    @classmethod
    def training_configuration_to_rest(cls, training_configuration: TrainingConfiguration) -> dict[str, Any]:
        """
        Convert a training configuration to its REST representation.

        Also supports PartialTrainingConfiguration objects.

        Args:
            training_configuration (TrainingConfiguration): The training configuration to convert.

        Returns:
            dict[str, Any]: REST representation of the training configuration.

        Raises:
            ValueError: If model_manifest_id is not set in the training configuration.
        """
        if not training_configuration.model_manifest_id:
            raise ValueError("Model manifest ID is required to convert training configuration to REST view")

        model_manifest = SupportedModels.get_model_manifest_by_id(training_configuration.model_manifest_id)
        default_config = model_manifest.hyperparameters.model_dump()

        training_params_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=training_configuration.hyperparameters.training,
                default_config=default_config.get("training", {}),
            )
            if training_configuration.hyperparameters and training_configuration.hyperparameters.training
            else []
        )

        return {
            "model_manifest_id": training_configuration.model_manifest_id,
            DATASET_PREPARATION: cls._dataset_preparation_to_rest(
                global_parameters=training_configuration.global_parameters,
                hyperparameters=training_configuration.hyperparameters,
                default_config=default_config,
            ),
            TRAINING: training_params_rest,
            EVALUATION: [],  # Evaluation parameters are not yet available
        }

    @classmethod
    def training_configuration_from_rest(cls, rest_input: dict[str, Any]) -> PartialTrainingConfiguration:
        """
        Convert REST input to a PartialTrainingConfiguration object.

        Parses REST API input and constructs a partial training configuration
        by distributing parameters between global parameters and hyperparameters.

        Args:
            rest_input (dict[str, Any]): REST input dictionary containing configuration data.

        Returns:
            PartialTrainingConfiguration: Validated partial training configuration object.
        """
        dataset_preparation = cls.configurable_parameters_from_rest(rest_input.pop(DATASET_PREPARATION, {}))
        training = cls.configurable_parameters_from_rest(rest_input.pop(TRAINING, {}))
        evaluation = cls.configurable_parameters_from_rest(rest_input.pop(EVALUATION, {}))

        global_parameters: dict = defaultdict(dict)
        hyperparameters: dict = defaultdict(dict)

        for field, _ in GlobalDatasetPreparationParameters.model_fields.items():
            global_parameters[DATASET_PREPARATION][field] = dataset_preparation.pop(field, None)

        for field, _ in DatasetPreparationParameters.model_fields.items():
            hyperparameters[DATASET_PREPARATION][field] = dataset_preparation.pop(field, None)

        for field, _ in TrainingHyperParameters.model_fields.items():
            hyperparameters[TRAINING][field] = training.pop(field, None)

        for field, _ in EvaluationParameters.model_fields.items():
            hyperparameters[EVALUATION][field] = evaluation.pop(field, None)

        # add remaining parameters for validation (extra parameters should not be present)
        global_parameters[DATASET_PREPARATION].update(dataset_preparation)
        hyperparameters[TRAINING].update(training)
        hyperparameters[EVALUATION].update(evaluation)

        # Convert defaultdict to regular dicts for the model validation
        global_parameters = dict(global_parameters)
        hyperparameters = dict(hyperparameters)
        global_parameters.pop("default_factory", None)
        hyperparameters.pop("default_factory", None)

        dict_model = {
            "global_parameters": global_parameters,
            "hyperparameters": hyperparameters,
        }

        return PartialTrainingConfiguration.model_validate(dict_model | rest_input)
