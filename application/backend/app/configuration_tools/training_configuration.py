# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator

from .parameters import Filtering, GlobalParameters, Hyperparameters, SubsetSplit
from .utils import partial_model


class TrainingConfiguration(BaseModel):
    """Configuration for model training"""

    model_manifest_id: str | None = Field(
        default=None,
        title="Model manifest ID",
        description="ID for the model manifest that defines the supported parameters and capabilities for training",
    )
    global_parameters: GlobalParameters = Field(
        title="Global parameters", description="Global configuration parameters for training"
    )
    hyperparameters: Hyperparameters = Field(title="Hyperparameters", description="Hyperparameters for training")

    @staticmethod
    def from_manifest(model_architecture_id: str, hyperparameters: Hyperparameters) -> "TrainingConfiguration":
        """
        Update the TrainingConfiguration with new hyperparameters.

        :param model_architecture_id: The model architect ID associated with the configuration
        :param hyperparameters: New hyperparameters to set
        :return: Updated TrainingConfiguration instance
        """
        return TrainingConfiguration(
            model_manifest_id=model_architecture_id,
            global_parameters=GlobalParameters(
                subset_split=SubsetSplit(),
                filtering=Filtering(),
            ),
            hyperparameters=hyperparameters,
        )

    def __eq__(self, other: object) -> bool:
        """
        Compares two ProjectConfiguration instances.

        Checks if both objects have the same ID and task configurations.
        """
        if not isinstance(other, TrainingConfiguration):
            return False

        # Compare model manifest IDs
        if self.model_manifest_id != other.model_manifest_id:
            return False

        # Compare parameters
        return self.global_parameters == other.global_parameters and self.hyperparameters == other.hyperparameters


@partial_model
class PartialTrainingConfiguration(TrainingConfiguration):
    """
    A partial version of `TrainingConfiguration` with all fields optional.

    Enables flexible updates and partial validation, making it suitable for scenarios
    where only a subset of the configuration needs to be specified or changed.
    """

    @model_validator(mode="after")
    def validate_identifiers(self) -> "PartialTrainingConfiguration":
        if not self.task_id:
            raise ValueError("task_id must be provided in the configuration.")
        return self


@partial_model
class PartialGlobalParameters(GlobalParameters):
    """
    A partial version of `GlobalParameters` with all fields optional.

    Enables flexible updates and partial validation, making it suitable for scenarios
    where only a subset of the configuration needs to be specified or changed.
    """
