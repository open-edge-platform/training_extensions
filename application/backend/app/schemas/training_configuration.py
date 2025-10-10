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
        """Create TrainingConfiguration from a ModelManifest's Hyperparameters"""
        return cls(
            dataset_preparation=hyperparameters.dataset_preparation,
            training=hyperparameters.training,
            evaluation=hyperparameters.evaluation,
        )

    @classmethod
    def from_model(cls, model_config: dict) -> "TrainingConfiguration":
        """Create TrainingConfiguration from model configuration dictionary"""
        return cls.model_validate(model_config)

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_preparation": {
                    "augmentation": {
                        "topdown_affine": None,
                        "random_zoom_out": None,
                        "iou_random_crop": {"enable": True},
                        "mosaic": None,
                        "random_resize_crop": None,
                        "random_affine": {
                            "enable": False,
                            "max_rotate_degree": 10,
                            "max_translate_ratio": 0.1,
                            "scaling_ratio_range": [0.5, 1.5],
                            "max_shear_degree": 2,
                        },
                        "mixup": None,
                        "hsv_random_aug": None,
                        "random_horizontal_flip": {"enable": True, "probability": 0.5},
                        "random_vertical_flip": {"enable": False, "probability": 0.5},
                        "color_jitter": {
                            "enable": False,
                            "brightness": [0.875, 1.125],
                            "contrast": [0.5, 1.5],
                            "saturation": [0.5, 1.5],
                            "hue": [-0.05, 0.05],
                            "probability": 0.5,
                        },
                        "gaussian_blur": {"enable": False, "kernel_size": 5, "sigma": [0.1, 2], "probability": 0.5},
                        "photometric_distort": None,
                        "gaussian_noise": {"enable": False, "mean": 0, "sigma": 0.1, "probability": 0.5},
                        "tiling": {"enable": False, "adaptive_tiling": True, "tile_size": 400, "tile_overlap": 0.2},
                    }
                },
                "training": {
                    "max_epochs": 200,
                    "early_stopping": {"enable": True, "patience": 10},
                    "learning_rate": 0.004,
                    "input_size_width": 992,
                    "input_size_height": 800,
                    "allowed_values_input_size": [128, 256, 384, 512, 640, 800, 992, 1024],
                },
                "evaluation": {"metric": "f_measure"},
            }
        }
    }
