#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator

from .augmentation import AugmentationParameters


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

    @model_validator(mode="after")
    def validate_subsets(self) -> "SubsetSplit":
        if (self.training + self.validation + self.test) != 100:
            raise ValueError("Sum of subsets should be equal to 100")
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

    min_annotation_pixels: MinAnnotationPixels = Field(
        default_factory=MinAnnotationPixels,
        title="Minimum annotation pixels",
        description="Minimum number of pixels in an annotation",
    )
    min_annotation_objects: MinAnnotationObjects = Field(
        default_factory=MinAnnotationObjects,
        title="Minimum annotation objects",
        description="Minimum number of objects in an annotation",
    )
    max_annotation_objects: MaxAnnotationObjects = Field(
        default_factory=MaxAnnotationObjects,
        title="Maximum annotation objects",
        description="Maximum number of objects in an annotation",
    )


class TaskLevelDatasetPreparationParameters(BaseModel):
    """
    Dataset preparation parameters that apply at the task level, for all models regardless of their specific
    architecture.

    These parameters control how the dataset is prepared before being passed to the training backend,
    including how it is split into subsets and how annotations are filtered based on certain criteria.
    """

    subset_split: SubsetSplit = Field(
        default_factory=SubsetSplit,
        title="Subset split",
        description=(
            "Subset split parameters define how the dataset is divided into training, validation, and test subsets. "
            "The training subset is used to fit the model, the validation subset is used to estimate the prediction "
            "error during training and the test subset is used to evaluate the final performance of the model. "
            "The percentages for training, validation, and test subsets must sum to 100."
        ),
    )
    filtering: Filtering = Field(
        default_factory=Filtering,
        title="Filtering",
        description=(
            "Filtering parameters define criteria for including or excluding annotations from the dataset. "
            "Depending on the scenario, an appropriate filter configuration can speed up the training process and/or "
            "improve the model performance by removing noisy annotations."
        ),
    )


class AlgoLevelDatasetPreparationParameters(BaseModel):
    """Dataset preparation parameters that depend on the specific algorithm being trained."""

    augmentation: AugmentationParameters = Field(
        default_factory=AugmentationParameters,
        title="Data augmentation",
        description=(
            "Data augmentation is a technique used in machine learning to artificially expand a training dataset "
            "by applying transformations (e.g., rotation, scaling, noise) to existing data. "
            "It improves model generalization and reduces overfitting by increasing data variability "
            "without collecting new samples."
        ),
    )
