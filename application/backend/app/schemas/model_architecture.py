# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from app.supported_models.model_manifest import ModelManifest


class ModelArchitectures(BaseModel):
    """Model architectures response"""

    model_architectures: list[ModelManifest] = Field(
        title="Model Architectures", description="List of available model architectures"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_architectures": [
                    {
                        "id": "Object_Detection_Deim_DFine_M",
                        "name": "Deim-DFine-M",
                        "description": "DEIM is an advanced training framework designed to enhance the matching"
                        " mechanism in DETRs, enabling faster convergence and improved accuracy.",
                        "task": "detection",
                        "stats": {
                            "gigaflops": 57,
                            "trainable_parameters": 19,
                            "performance_ratings": {"accuracy": 2, "training_time": 2, "inference_speed": 2},
                        },
                        "support_status": "active",
                        "supported_gpus": {"intel": True, "nvidia": True},
                        "capabilities": {"xai": True, "tiling": True},
                        "hyperparameters": {
                            "dataset_preparation": {
                                "augmentation": {
                                    "topdown_affine": None,
                                    "random_zoom_out": None,
                                    "iou_random_crop": None,
                                    "mosaic": None,
                                    "random_resize_crop": None,
                                    "random_affine": None,
                                    "mixup": None,
                                    "hsv_random_aug": None,
                                    "random_horizontal_flip": None,
                                    "random_vertical_flip": None,
                                    "color_jitter": None,
                                    "gaussian_blur": None,
                                    "photometric_distort": None,
                                    "gaussian_noise": None,
                                    "tiling": {
                                        "enable": False,
                                        "adaptive_tiling": True,
                                        "tile_size": 400,
                                        "tile_overlap": 0.2,
                                    },
                                }
                            },
                            "training": {
                                "max_epochs": 200,
                                "early_stopping": {"enable": True, "patience": 10},
                                "learning_rate": 0.0004,
                                "input_size_width": 640,
                                "input_size_height": 640,
                                "allowed_values_input_size": [640],
                            },
                            "evaluation": {"metric": "f_measure"},
                        },
                        "is_default_model": False,
                        "model_category": None,
                    },
                ]
            }
        }
    }
