# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, model_validator

from .augmentation import AugmentationParameters
from .base_model_no_extra import BaseModelNoExtra
from .utils import partial_model


class DatasetPreparationParameters(BaseModelNoExtra):
    """Parameters for dataset preparation before training."""

    augmentation: AugmentationParameters = Field(
        default_factory=AugmentationParameters,
        title="Data augmentation",
        description="Configuration for data augmentation techniques applied to the dataset",
    )


class EarlyStopping(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable early stopping",
        description="Whether to stop training early when performance stops improving",
    )
    patience: int = Field(
        gt=0,
        default=1,
        title="Patience",
        description="Number of epochs with no improvement after which training will be stopped",
    )


class TrainingHyperParameters(BaseModelNoExtra):
    """Hyperparameters for model training process."""

    max_epochs: int | None = Field(
        gt=0, default=None, title="Maximum epochs", description="Maximum number of training epochs to run"
    )
    early_stopping: EarlyStopping | None = Field(
        default=None, title="Early stopping", description="Configuration for early stopping mechanism"
    )
    learning_rate: float | None = Field(
        gt=0, lt=1, default=None, title="Learning rate", description="Base learning rate for the optimizer"
    )
    input_size_width: int | None = Field(
        default=None,
        gt=0,
        title="Input size width",
        description=(
            "Width dimension in pixels for model input images. "
            "Determines the horizontal resolution at which images are processed."
        ),
        json_schema_extra={},
    )
    input_size_height: int | None = Field(
        default=None,
        gt=0,
        title="Input size height",
        description=(
            "Height dimension in pixels for model input images. "
            "Determines the vertical resolution at which images are processed."
        ),
        json_schema_extra={},
    )
    allowed_values_input_size: list[int] | None = Field(
        default=None,
        title="Supported input size dimensions",
        description=(
            "List of supported values for input width and height. "
            "When specified, both width and height must be chosen from these values."
        ),
        json_schema_extra={"validation_only": True},
    )

    @model_validator(mode="after")
    def validate_input_size(self) -> "TrainingHyperParameters":
        w, h = self.input_size_width, self.input_size_height

        # Skip validation if neither width nor height are set
        if w is None and h is None:
            return self

        # For non-partial models, both width and height must be set
        class_name = type(self).__name__
        if "partial" not in class_name.lower() and (w is None or h is None):
            raise ValueError("Both input_size_width and input_size_height must be specified")

        # Validate against allowed input sizes if available
        if allowed_values := self.allowed_values_input_size:
            if w and w not in allowed_values:
                raise ValueError(
                    f"Input size width '{w}' is not in the list of supported input sizes: {allowed_values}"
                )
            if h and h not in allowed_values:
                raise ValueError(
                    f"Input size height '{h}' is not in the list of supported input sizes: {allowed_values}"
                )

        # Update the model's json_schema to include allowed values for input_size
        # Note: existing json_schema_extra cannot be overwritten, otherwise it will absent from model_json_schema()
        self.model_fields["input_size_width"].json_schema_extra.update(  # type: ignore[union-attr]
            {
                "allowed_values": self.allowed_values_input_size,  # type:ignore[dict-item]
                "default_value": w,
            }
        )
        self.model_fields["input_size_height"].json_schema_extra.update(  # type: ignore[union-attr]
            {
                "allowed_values": self.allowed_values_input_size,  # type:ignore[dict-item]
                "default_value": h,
            }
        )
        return self


class EvaluationParameters(BaseModelNoExtra):
    """Parameters for model evaluation."""

    metric: str | None = Field(
        default=None, title="Evaluation metric", description="Metric used to evaluate model performance"
    )


class Hyperparameters(BaseModelNoExtra):
    """Complete set of configurable parameters for model training and evaluation."""

    dataset_preparation: DatasetPreparationParameters = Field(
        default_factory=DatasetPreparationParameters,
        title="Dataset preparation",
        description="Parameters for preparing the dataset before training",
    )
    training: TrainingHyperParameters | None = Field(
        default=None,
        title="Training hyperparameters",
        description="Hyperparameters for the model training process",
    )
    evaluation: EvaluationParameters = Field(
        default_factory=EvaluationParameters,
        title="Evaluation parameters",
        description="Parameters for evaluating the trained model",
    )


@partial_model
class PartialHyperparameters(Hyperparameters):
    """
    A partial version of `Hyperparameters` with all fields optional.

    Enables flexible updates and partial validation, making it suitable for scenarios
    where only a subset of the configuration needs to be specified or changed.
    """
