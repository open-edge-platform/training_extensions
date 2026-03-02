# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator


class EarlyStopping(BaseModel):
    enable: bool = Field(
        default=False,
        title="Toggle early stopping",
        description="Whether to stop training early when performance stops improving",
    )
    patience: int = Field(
        gt=0,
        default=1,
        title="Patience",
        description="Number of epochs with no improvement after which training will be stopped",
    )


class AlgoLevelTrainingParameters(BaseModel):
    """Hyperparameters for model training process."""

    max_epochs: int = Field(
        gt=0,
        default=200,
        title="Maximum epochs",
        description=(
            "Maximum number of epochs to train the model. An epoch is one complete pass through the training dataset."
        ),
    )
    early_stopping: EarlyStopping = Field(
        default_factory=EarlyStopping,
        title="Early stopping",
        description=(
            "Early stopping is a technique to prevent overfitting by stopping training when performance "
            "on a validation set stops improving."
        ),
    )
    learning_rate: float = Field(
        gt=0,
        lt=1,
        title="Learning rate",
        description=(
            "Learning rate for the optimizer, controlling the step size during model weight updates. "
            "A smaller learning rate may lead to more stable convergence, while a larger learning rate may speed up "
            "training but risk overshooting minima in the loss landscape."
        ),
    )
    input_size_width: int = Field(
        gt=0,
        title="Input size width",
        description=(
            "Width size in pixels for model input images. "
            "Determines the horizontal resolution at which images are processed."
        ),
        json_schema_extra={"allowed_values_from": "allowed_values_input_size"},
    )
    input_size_height: int = Field(
        gt=0,
        title="Input size height",
        description=(
            "Height size in pixels for model input images. "
            "Determines the vertical resolution at which images are processed."
        ),
        json_schema_extra={"allowed_values_from": "allowed_values_input_size"},
    )
    allowed_values_input_size: list[int] = Field(
        title="Supported input size (width and height)",
        description="List of supported values for input width and height.",
        json_schema_extra={"validation_only": True},
    )

    @model_validator(mode="after")
    def validate_input_size(self) -> "AlgoLevelTrainingParameters":
        # Validate that allowed_values_input_size is provided and not empty
        if len(self.allowed_values_input_size) == 0:
            raise ValueError("'allowed_values_input_size' must contain at least one value")

        # Validate that input_size_width and input_size_height are specified
        w, h = self.input_size_width, self.input_size_height
        if w is None or h is None:
            raise ValueError("Both 'input_size_width' and 'input_size_height' must be specified")

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
        return self
