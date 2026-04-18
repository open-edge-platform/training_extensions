#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .augmentation import AugmentationParameters


class IntensityMappingMode(StrEnum):
    """Mode for mapping image intensity values before training."""

    SCALE_TO_UNIT = "Unit interval scaling"  # (x / max_value)
    WINDOW = "Windowing"  # clamp((x - low) / (high - low), 0, 1)
    RANGE_SCALE = "Clipped scaling"  # clamp(x * factor, min_value, max_value)


class IntensityMapping(BaseModel):
    """Parameters for mapping image intensity values before training.

    Intensity mapping is important when working with images whose pixel range differs from the standard
    8-bit [0, 255] range, for example 16-bit images commonly found in medical or scientific imaging.
    """

    mode: IntensityMappingMode = Field(
        default=IntensityMappingMode.SCALE_TO_UNIT,
        title="Intensity mapping mode",
        description=(
            "Strategy used to transform pixel intensities. "
            "'Unit interval scaling' divides by max_value, thus mapping the range [0, max_value] to [0, 1]. "
            "'Windowing' isolates a specific intensity range, mapping a specific window (specified with center and "
            "width) to [0, 1] and clipping values outside the window. "
            "'Clipped scaling' multiplies pixel values by a scale factor and clips the result to a specified range "
            "(min_value, max_value)."
        ),
    )
    max_value: float = Field(
        default=255.0,
        ge=0.0,
        title="Maximum pixel value",
        description=(
            "Maximum possible pixel value in the raw image. For 8-bit images use 255, for 16-bit images use 65535."
        ),
    )
    min_value: float = Field(
        default=0.0,
        title="Minimum output value",
        description=("Minimum output value after rescaling the image; Pixel values below this threshold are clipped."),
        json_schema_extra={"depends_on": {"mode": "Clipped scaling"}},
    )
    window_center: float = Field(
        default=127.5,
        title="Window center",
        description=(
            "Center of the intensity window for 'windowing' mode. "
            "Together with width, it defines the intensity range that is mapped to [0, 1]."
        ),
        json_schema_extra={"depends_on": {"mode": "Windowing"}},
    )
    window_width: float = Field(
        default=255.0,
        gt=0.0,
        title="Window width",
        description=(
            "Width of the intensity window for 'windowing' mode. "
            "The effective range is [center - width/2, center + width/2], mapped linearly to [0, 1]."
        ),
        json_schema_extra={"depends_on": {"mode": "Windowing"}},
    )
    scale_factor: float = Field(
        default=1.0,
        gt=0.0,
        title="Scale factor",
        description=(
            "Multiplicative factor applied to pixel values, before clipping the result to [min_value, max_value]."
        ),
        json_schema_extra={"depends_on": {"mode": "Clipped scaling"}},
    )


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
    intensity_mapping: IntensityMapping = Field(
        default_factory=IntensityMapping,
        title="Intensity mapping",
        description=(
            "Intensity mapping parameters control how raw pixel values are normalised before training. "
            "This is especially important for images with non-standard bit depths (e.g. 16-bit), where the "
            "default [0, 255] assumption does not hold."
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
