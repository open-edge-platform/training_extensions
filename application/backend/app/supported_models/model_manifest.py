# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from enum import Enum
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.core.models import TaskType

from .default_models import DefaultCategory, DefaultModels
from .hyperparameters import Hyperparameters


class GPUMaker(str, Enum):
    """GPU maker names."""

    NVIDIA = "nvidia"
    INTEL = "intel"

    def __str__(self) -> str:
        """Returns the name of the GPU maker."""
        return str(self.name)


class ModelManifestDeprecationStatus(str, Enum):
    """Status of a model architecture with respect to the deprecation process."""

    ACTIVE = "active"  # Model architecture is fully supported, models can be trained
    DEPRECATED = "deprecated"  # Model architecture is deprecated, can still view and train but it's discouraged
    OBSOLETE = "obsolete"  # Model architecture is no longer supported, models can be still viewed but not trained

    def __str__(self) -> str:
        """Returns the name of the model status."""
        return str(self.name)


class PerformanceRatings(BaseModel):
    """Ratings for different performance aspects of a model."""

    model_config = ConfigDict(extra="forbid")
    accuracy: int = Field(
        ge=1,
        le=3,
        default=1,
        title="Accuracy rating",
        description="Rating of the model accuracy. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average) to 3 (above average).",
    )
    training_time: int = Field(
        ge=1,
        le=3,
        default=1,
        title="Training time rating",
        description="Rating of the model training time. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average/slower) to 3 (above average/faster).",
    )
    inference_speed: int = Field(
        ge=1,
        le=3,
        default=1,
        title="Inference speed rating",
        description="Rating of the model inference speed. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average/slower) to 3 (above average/faster).",
    )


class ModelStats(BaseModel):
    """Information about a machine learning model."""

    model_config = ConfigDict(extra="forbid")
    gigaflops: float = Field(
        ge=0, title="Gigaflops", description="Billions of floating-point operations per second required by the model"
    )
    trainable_parameters: float = Field(
        ge=0.0,
        default=0.0,
        title="Trainable parameters (millions)",
        description="Number of trainable parameters in the model, expressed in millions",
    )
    performance_ratings: PerformanceRatings = Field(
        title="Performance ratings", description="Standardized ratings for model performance metrics"
    )


class Capabilities(BaseModel):
    """Model capabilities configuration."""

    model_config = ConfigDict(extra="forbid")
    xai: bool = Field(
        default=False, title="Explainable AI Support", description="Whether the model supports explainable AI features"
    )
    tiling: bool = Field(
        default=False,
        title="Tiling Support",
        description="Whether the model supports image tiling for processing large images",
    )


class PretrainedWeights(BaseModel):
    """Pretrained weights information."""

    model_config = ConfigDict(extra="forbid")
    url: str = Field(title="Weights URL", description="URL to download the pretrained weights")
    sha_sum: str = Field(title="Weights SHA256", description="SHA256 checksum of the pretrained weights file")


class ModelManifest(BaseModel):
    """ModelManifest contains the necessary information for training a specific machine learning model."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(title="Model architecture ID", description="Unique identifier for the model architecture")
    name: str = Field(title="Model architecture name", description="Friendly name of the model architecture")
    pretrained_weights: PretrainedWeights = Field(
        title="Pretrained Weights", description="URL and SHA sum of the pretrained weights"
    )
    description: str = Field(title="Description", description="Detailed description of the model capabilities")
    task: TaskType = Field(title="Task Type", description="Type of machine learning task addressed by the model")
    stats: ModelStats = Field(title="Model Statistics", description="Statistics about the model")
    support_status: ModelManifestDeprecationStatus = Field(
        default=ModelManifestDeprecationStatus.ACTIVE,
        title="Support Status",
        description="Current support level (active, deprecated, or obsolete)",
    )
    supported_gpus: dict[GPUMaker, bool] = Field(
        title="Supported GPUs", description="Dictionary mapping GPU types to compatibility status"
    )
    capabilities: Capabilities = Field(
        title="Model Capabilities", description="Special capabilities supported by the model"
    )
    hyperparameters: Hyperparameters = Field(
        title="Hyperparameters", description="Configuration parameters for model training"
    )

    @computed_field  # type: ignore[misc]
    @cached_property
    def is_default_model(self) -> bool:
        """Returns whether this model is the default one for its task type"""
        return DefaultModels.get_default_model(self.task) == self.id

    @computed_field  # type: ignore[misc]
    @cached_property
    def model_category(self) -> str | None:
        """Returns the category for which this model is recommended (accuracy, speed, or balance)"""
        if DefaultModels.get_accuracy_model(self.task) == self.id:
            return DefaultCategory.ACCURACY.name.lower()
        if DefaultModels.get_speed_model(self.task) == self.id:
            return DefaultCategory.SPEED.name.lower()
        if DefaultModels.get_balanced_model(self.task) == self.id:
            return DefaultCategory.BALANCE.name.lower()
        return None
