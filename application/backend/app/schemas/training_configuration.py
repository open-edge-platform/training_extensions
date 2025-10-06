# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from app.supported_models.hyperparameters import (
    DatasetPreparationParameters,
    EvaluationParameters,
    Hyperparameters,
    TrainingHyperParameters,
)


class TrainingConfiguration(BaseModel):
    """Complete training configuration schema."""

    dataset_preparation: DatasetPreparationParameters = Field(
        title="Dataset preparation", description="Parameters for dataset preparation before training"
    )
    training: TrainingHyperParameters | None = Field(
        title="Training hyperparameters", description="Hyperparameters for model training process"
    )
    evaluation: EvaluationParameters = Field(
        title="Evaluation parameters", description="Parameters for evaluating the trained model"
    )

    @classmethod
    def from_hyperparameters(cls, hyperparameters: Hyperparameters) -> "TrainingConfiguration":
        """Create TrainingConfiguration from Hyperparameters"""
        return cls(
            dataset_preparation=hyperparameters.dataset_preparation,
            training=hyperparameters.training,
            evaluation=hyperparameters.evaluation,
        )

    # TODO: Add example values #4799
    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_preparation": {"config": {"train_val_split_ratio": 0.8, "augmentation_enabled": True}},
                "training": {"config": {"epochs": 100, "batch_size": 32, "learning_rate": 0.001}},
                "evaluation": {"config": {"metrics": ["accuracy", "precision", "recall"], "validation_split": 0.2}},
            }
        }
    }
