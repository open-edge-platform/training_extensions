# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator

from app.models.partial import partial_model
from app.models.training_configuration.hyperparameters import Hyperparameters


class SubsetSplit(BaseModel):
    """
    Parameters for splitting a dataset into training, validation, and test subsets.
    The sum of training, validation, and test percentages must equal 100.
    """

    training: int = Field(
        ge=1, le=100, default=70, title="Training percentage", description="Percentage of data to use for training"
    )
    validation: int = Field(
        ge=1, le=100, default=20, title="Validation percentage", description="Percentage of data to use for validation"
    )
    test: int = Field(
        ge=1, le=100, default=10, title="Test percentage", description="Percentage of data to use for testing"
    )
    auto_selection: bool = Field(
        default=False, title="Auto selection", description="Whether to automatically select data for each subset"
    )
    remixing: bool = Field(default=False, title="Remixing", description="Whether to remix data between subsets")
    dataset_size: int | None = Field(
        ge=0,
        default=None,
        title="Dataset size",
        description="Total size of the dataset (read-only parameter, not configurable by users)",
        exclude=True,  # exclude read-only parameter from serialization
        json_schema_extra={"default_value": None},
    )

    @model_validator(mode="after")
    def validate_subsets(self) -> "SubsetSplit":
        if (self.training + self.validation + self.test) != 100:
            raise ValueError("Sum of subsets should be equal to 100")
        # check that all subsets can have at least one item
        if self.dataset_size is not None and self.dataset_size < 3:
            raise ValueError("The dataset is too small to assign at least one item to each subset")
        return self


class MinAnnotationPixels(BaseModel):
    """Parameters for minimum annotation pixels."""

    enable: bool = Field(
        default=False,
        title="Enable minimum annotation pixels filtering",
        description="Whether to apply minimum annotation pixels filtering",
    )
    value: int = Field(
        gt=0,
        le=200000000,  # reasonable upper limit for pixel count to 200MP
        default=1,
        title="Minimum annotation pixels",
        description="Minimum number of pixels in an annotation",
    )


class MaxAnnotationPixels(BaseModel):
    """Parameters for maximum annotation pixels."""

    enable: bool = Field(
        default=False,
        title="Enable maximum annotation pixels filtering",
        description="Whether to apply maximum annotation pixels filtering",
    )
    value: int = Field(
        gt=0, default=10000, title="Maximum annotation pixels", description="Maximum number of pixels in an annotation"
    )


class MinAnnotationObjects(BaseModel):
    """Parameters for maximum annotation objects."""

    enable: bool = Field(
        default=False,
        title="Enable minimum annotation objects filtering",
        description="Whether to apply minimum annotation objects filtering",
    )
    value: int = Field(
        gt=0,
        default=1,
        title="Minimum annotation objects",
        description="Minimum number of objects in an annotation",
    )


class MaxAnnotationObjects(BaseModel):
    """Parameters for maximum annotation objects."""

    enable: bool = Field(
        default=False,
        title="Enable maximum annotation objects filtering",
        description="Whether to apply maximum annotation objects filtering",
    )
    value: int = Field(
        gt=0,
        default=10000,
        title="Maximum annotation objects",
        description="Maximum number of objects in an annotation",
    )


class Filtering(BaseModel):
    """Parameters for filtering annotations in the dataset."""

    min_annotation_pixels: MinAnnotationPixels | None = Field(
        default=None, title="Minimum annotation pixels", description="Minimum number of pixels in an annotation"
    )
    max_annotation_pixels: MaxAnnotationPixels | None = Field(
        default=None, title="Maximum annotation pixels", description="Maximum number of pixels in an annotation"
    )
    min_annotation_objects: MinAnnotationObjects | None = Field(
        default=None, title="Minimum annotation objects", description="Minimum number of objects in an annotation"
    )
    max_annotation_objects: MaxAnnotationObjects | None = Field(
        default=None, title="Maximum annotation objects", description="Maximum number of objects in an annotation"
    )


class GlobalDatasetPreparationParameters(BaseModel):
    """
    Parameters for preparing a dataset for training within the global configuration.
    Controls data splitting and filtering before being passed for the training.
    """

    subset_split: SubsetSplit = Field(title="Subset split", description="Configuration for splitting data into subsets")
    filtering: Filtering = Field(
        default_factory=Filtering, title="Filtering", description="Configuration for filtering annotations"
    )


class GlobalParameters(BaseModel):
    """
    Global parameters that are used within the application but are not directly passed to the training backend.
    These parameters still impact the final training outcome by controlling dataset preparation.
    """

    dataset_preparation: GlobalDatasetPreparationParameters = Field(
        title="Dataset preparation", description="Parameters for preparing the dataset"
    )


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


@partial_model
class PartialTrainingConfiguration(TrainingConfiguration):
    """
    A partial version of `TrainingConfiguration` with all fields optional.

    Enables flexible updates and partial validation, making it suitable for scenarios
    where only a subset of the configuration needs to be specified or changed.
    """


@partial_model
class PartialGlobalParameters(GlobalParameters):
    """
    A partial version of `GlobalParameters` with all fields optional.

    Enables flexible updates and partial validation, making it suitable for scenarios
    where only a subset of the configuration needs to be specified or changed.
    """
