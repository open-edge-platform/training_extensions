# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from app.models.training_configuration import (
    AugmentationParameters,
    ColorJitter,
    DatasetPreparationParameters,
    EarlyStopping,
    EvaluationParameters,
    GaussianBlur,
    Hyperparameters,
    PartialHyperparameters,
    RandomAffine,
    RandomHorizontalFlip,
    RandomIOUCrop,
    RandomResizeCrop,
    RandomVerticalFlip,
    Tiling,
    TrainingHyperParameters,
)


class TestHyperparameters:
    @pytest.mark.parametrize(
        "hyperparams_dict, expected_params",
        [
            # Test case 1: Basic configuration
            (
                {
                    "dataset_preparation": {"augmentation": {"random_horizontal_flip": {"enable": True}}},
                    "training": {
                        "max_epochs": 50,
                        "learning_rate": 0.01,
                        "input_size_width": 32,
                        "input_size_height": 32,
                        "allowed_values_input_size": [32],
                    },
                    "evaluation": {},
                },
                Hyperparameters(
                    dataset_preparation=DatasetPreparationParameters(
                        augmentation=AugmentationParameters(random_horizontal_flip=RandomHorizontalFlip(enable=True))
                    ),
                    training=TrainingHyperParameters(
                        max_epochs=50,
                        learning_rate=0.01,
                        input_size_width=32,
                        input_size_height=32,
                        allowed_values_input_size=[32],
                    ),
                    evaluation=EvaluationParameters(),
                ),
            ),
            # Test case 2: Complex configuration with all augmentation options
            (
                {
                    "dataset_preparation": {
                        "augmentation": {
                            "random_resize_crop": {
                                "enable": True,
                                "crop_ratio_range": [0.2, 1.0],
                                "aspect_ratio_range": [0.5, 2.0],
                            },
                            "random_affine": {
                                "enable": True,
                                "max_rotate_degree": 30.0,
                                "max_translate_ratio": 0.1,
                                "scaling_ratio_range": [0.5, 1.5],
                                "max_shear_degree": 2.0,
                            },
                            "random_horizontal_flip": {"enable": True, "probability": 0.5},
                            "random_vertical_flip": {"enable": True, "probability": 0.5},
                            "iou_random_crop": {"enable": True},
                            "color_jitter": {
                                "enable": True,
                                "brightness": [0.875, 1.125],
                                "contrast": [0.5, 1.5],
                                "saturation": [0.5, 1.5],
                                "hue": [-0.05, 0.05],
                                "probability": 0.5,
                            },
                            "gaussian_blur": {
                                "enable": True,
                                "kernel_size": 3,
                                "sigma": [0.1, 2.0],
                                "probability": 0.5,
                            },
                            "tiling": {"enable": True, "adaptive_tiling": True, "tile_size": 224, "tile_overlap": 0.15},
                        }
                    },
                    "training": {
                        "max_epochs": 100,
                        "learning_rate": 0.001,
                        "early_stopping": {"enable": True, "patience": 10},
                        "input_size_width": 32,
                        "input_size_height": 64,
                        "allowed_values_input_size": [32, 64, 128],
                    },
                    "evaluation": {},
                },
                Hyperparameters(
                    dataset_preparation=DatasetPreparationParameters(
                        augmentation=AugmentationParameters(
                            random_resize_crop=RandomResizeCrop(
                                enable=True, crop_ratio_range=[0.2, 1.0], aspect_ratio_range=[0.5, 2.0]
                            ),
                            random_affine=RandomAffine(
                                enable=True,
                                max_rotate_degree=30.0,
                                max_translate_ratio=0.1,
                                scaling_ratio_range=[0.5, 1.5],
                                max_shear_degree=2.0,
                            ),
                            random_horizontal_flip=RandomHorizontalFlip(enable=True, probability=0.5),
                            random_vertical_flip=RandomVerticalFlip(enable=True, probability=0.5),
                            iou_random_crop=RandomIOUCrop(enable=True),
                            color_jitter=ColorJitter(
                                enable=True,
                                brightness=[0.875, 1.125],
                                contrast=[0.5, 1.5],
                                saturation=[0.5, 1.5],
                                hue=[-0.05, 0.05],
                                probability=0.5,
                            ),
                            gaussian_blur=GaussianBlur(
                                enable=True,
                                kernel_size=3,
                                sigma=[0.1, 2.0],
                                probability=0.5,
                            ),
                            tiling=Tiling(enable=True, adaptive_tiling=True, tile_size=224, tile_overlap=0.15),
                        )
                    ),
                    training=TrainingHyperParameters(
                        max_epochs=100,
                        learning_rate=0.001,
                        early_stopping=EarlyStopping(enable=True, patience=10),
                        input_size_width=32,
                        input_size_height=64,
                        allowed_values_input_size=[32, 64, 128],
                    ),
                    evaluation=EvaluationParameters(),
                ),
            ),
            # Test case 3: Edge case values
            (
                {
                    "dataset_preparation": {
                        "augmentation": {
                            "random_resize_crop": {"enable": True, "crop_ratio_range": [0.1, 1.0]},
                            "random_horizontal_flip": {"enable": False},
                        }
                    },
                    "training": {
                        "max_epochs": 1,
                        "learning_rate": 0.0001,
                        "early_stopping": {"enable": True, "patience": 1},
                        "input_size_width": 32,
                        "input_size_height": 32,
                        "allowed_values_input_size": [32, 64, 128],
                    },
                    "evaluation": {},
                },
                Hyperparameters(
                    dataset_preparation=DatasetPreparationParameters(
                        augmentation=AugmentationParameters(
                            random_resize_crop=RandomResizeCrop(enable=True, crop_ratio_range=[0.1, 1.0]),
                            random_horizontal_flip=RandomHorizontalFlip(enable=False),
                        )
                    ),
                    training=TrainingHyperParameters(
                        max_epochs=1,
                        learning_rate=0.0001,
                        early_stopping=EarlyStopping(enable=True, patience=1),
                        input_size_width=32,
                        input_size_height=32,
                        allowed_values_input_size=[32, 64, 128],
                    ),
                    evaluation=EvaluationParameters(),
                ),
            ),
        ],
    )
    def test_valid_hyperparameters(self, hyperparams_dict: dict, expected_params: Hyperparameters) -> None:
        # Create model from dict
        params = Hyperparameters(**hyperparams_dict)

        # Compare with expected model
        assert params == expected_params

        # Validate specific fields to ensure proper parsing
        assert (
            params.dataset_preparation.augmentation.random_horizontal_flip
            == expected_params.dataset_preparation.augmentation.random_horizontal_flip
        )

        assert expected_params.training
        assert params.training
        assert params.training.early_stopping == expected_params.training.early_stopping
        assert params.training.max_epochs == expected_params.training.max_epochs
        assert params.training.learning_rate == expected_params.training.learning_rate

    @pytest.mark.parametrize(
        "hyperparams_dict",
        [
            # Test case 1: Invalid field types
            {
                "dataset_preparation": {},
                "training": {"max_epochs": "5o", "learning_rate": 0.01},
                "evaluation": {},
            },
            # Test case 2: Out of range values
            {
                "dataset_preparation": {"augmentation": {"random_horizontal_flip": {"enable": True}}},
                "training": {"max_epochs": -10, "learning_rate": 0.01},  # Negative max_epochs
                "evaluation": {},
            },
            # Test case 3: input_size has wrong format
            {
                "dataset_preparation": {},
                "training": {"input_size_width": "32x32", "input_size_height": "64x64"},
                "evaluation": {},
            },
            # Test case 4: input_size not in allowed sizes
            {
                "dataset_preparation": {},
                "training": {"input_size_width": 32, "input_size_height": 64, "allowed_values_input_size": [32]},
                "evaluation": {},
            },
            # Test case 5: input_size_width is set but input_size_height is not
            {
                "dataset_preparation": {},
                "training": {"input_size_width": 32, "allowed_values_input_size": [32]},
                "evaluation": {},
            },
        ],
        ids=[
            "Invalid field types (max_epochs as string)",
            "Out of range values (max_epochs < 0)",
            "input_size has wrong format (should be int)",
            "input_size not in allowed sizes (height not in allowed values)",
            "input_size_width set but input_size_height not set",
        ],
    )
    def test_validation_errors(self, hyperparams_dict) -> None:
        """Test that validation errors in nested models are properly caught"""
        with pytest.raises((ValidationError, ValueError)):
            Hyperparameters.model_validate(hyperparams_dict)

    def test_partial_hyperparameters(self) -> None:
        """Test that PartialHyperparameters works correctly with both partial and complete configurations."""
        # Test with a partial configuration
        partial_hyperparams = PartialHyperparameters.model_validate(
            {
                "training": {
                    "learning_rate": 0.005,
                    "early_stopping": {
                        "enable": True,
                    },
                    "input_size_width": 32,  # Partial model should allow input_size_height to be None
                }
            }
        )

        # Verify that specified fields are set correctly
        assert partial_hyperparams.training
        assert partial_hyperparams.training.learning_rate == 0.005
        assert partial_hyperparams.training.early_stopping
        assert partial_hyperparams.training.early_stopping.enable is True

        # Verify that unspecified fields are None
        assert partial_hyperparams.dataset_preparation is None
        assert partial_hyperparams.evaluation is None
        assert partial_hyperparams.training.max_epochs is None
        assert partial_hyperparams.training.early_stopping.patience is None
        assert partial_hyperparams.training.input_size_width == 32
        assert partial_hyperparams.training.input_size_height is None
        assert partial_hyperparams.training.allowed_values_input_size is None

        # Test with a nested partial configuration
        nested_partial_hyperparams = PartialHyperparameters.model_validate(
            {"dataset_preparation": {"augmentation": {"random_resize_crop": {"crop_ratio_range": [0.2, 0.75]}}}}
        )

        # Verify that nested fields are set correctly
        assert nested_partial_hyperparams.dataset_preparation.augmentation.random_resize_crop.crop_ratio_range == [
            0.2,
            0.75,
        ]
        assert nested_partial_hyperparams.dataset_preparation.augmentation.random_resize_crop.enable is None

        # Test with a full configuration
        full_hyperparams = Hyperparameters(
            dataset_preparation=DatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    random_resize_crop=RandomResizeCrop(enable=True, crop_ratio_range=[0.2, 0.6]),
                )
            ),
            training=TrainingHyperParameters(
                max_epochs=100,
                early_stopping=EarlyStopping(enable=True, patience=10),
                learning_rate=0.001,
                input_size_width=32,
                input_size_height=32,
                allowed_values_input_size=[32, 64, 128],
            ),
            evaluation=EvaluationParameters(),
        )

        full_hyperparams_dict = full_hyperparams.model_dump()
        partial_hyperparams_full = PartialHyperparameters.model_validate(full_hyperparams_dict)

        # Verify that the full configuration converted to partial is identical to the original
        assert partial_hyperparams_full.model_dump() == full_hyperparams.model_dump()

        # Verify that validation still works for partial models
        with pytest.raises(ValidationError):
            PartialHyperparameters.model_validate(
                {
                    "training": {
                        "learning_rate": -0.1  # Invalid: must be > 0
                    }
                }
            )
