#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import pytest

from app.api.schemas import TrainingConfigurationView
from app.models.training_configuration import (
    AlgoLevelDatasetPreparationParameters,
    AlgoLevelParameters,
    AlgoLevelTrainingParameters,
    TaskLevelDatasetPreparationParameters,
)
from app.models.training_configuration.augmentation import AugmentationParameters, ColorJitter
from app.models.training_configuration.configuration import TaskLevelParameters, TrainingConfiguration
from app.models.training_configuration.dataset_preparation import (
    Filtering,
    MaxAnnotationObjects,
    MinAnnotationObjects,
    MinAnnotationPixels,
    SubsetSplit,
)
from app.models.training_configuration.training import EarlyStopping


@pytest.fixture
def fxt_training_configuration() -> TrainingConfiguration:
    """Create a mock training configuration."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(
            dataset_preparation=TaskLevelDatasetPreparationParameters(
                subset_split=SubsetSplit(training=75, validation=15, test=10),
                filtering=Filtering(
                    min_annotation_pixels=MinAnnotationPixels(enable=True, value=20),
                    min_annotation_objects=MinAnnotationObjects(enable=False, value=1),
                    max_annotation_objects=MaxAnnotationObjects(enable=False, value=50),
                ),
            ),
        ),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    color_jitter=ColorJitter(
                        enable=True,
                        brightness=(0.9, 1.1),
                        contrast=(0.85, 1.15),
                        saturation=(0.8, 1.2),
                        hue=(-0.05, 0.05),
                        probability=0.6,
                    )
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=120,
                early_stopping=EarlyStopping(enable=True, patience=5),
                learning_rate=0.001,
                input_size_width=256,
                input_size_height=256,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_default_training_configuration() -> TrainingConfiguration:
    """Create a default training configuration with default values."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    color_jitter=ColorJitter(
                        enable=False,
                        brightness=(0.8, 1.2),
                        contrast=(0.75, 1.25),
                        saturation=(0.9, 1.1),
                        hue=(-0.1, 0.1),
                        probability=0.5,
                    )
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=250,
                early_stopping=EarlyStopping(enable=False, patience=1),
                learning_rate=0.0015,
                input_size_width=512,
                input_size_height=512,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_training_configuration_view_json() -> dict:
    return {
        "parameters": [
            {
                "type": "parameter_group",
                "key": "dataset_preparation",
                "name": "Dataset preparation",
                "description": (
                    "Configurable parameters related to the training data, such as augmentations and filters."
                ),
                "parameters": [
                    {
                        "type": "parameter_group",
                        "key": "subset_split",
                        "name": "Subset split",
                        "description": (
                            "Subset split parameters define how the dataset is divided into training, validation, "
                            "and test subsets. The training subset is used to fit the model, the validation subset "
                            "is used to estimate the prediction error during training and the test subset is used "
                            "to evaluate the final performance of the model. The percentages for training, validation, "
                            "and test subsets must sum to 100."
                        ),
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "training",
                                "name": "Training percentage",
                                "description": "Percentage of data to use for training",
                                "value": 75,
                                "default_value": 70,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": 100,
                                "allowed_values": None,
                            },
                            {
                                "type": "parameter",
                                "key": "validation",
                                "name": "Validation percentage",
                                "description": "Percentage of data to use for validation",
                                "value": 15,
                                "default_value": 20,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": 100,
                                "allowed_values": None,
                            },
                            {
                                "type": "parameter",
                                "key": "test",
                                "name": "Test percentage",
                                "description": "Percentage of data to use for testing",
                                "value": 10,
                                "default_value": 10,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": 100,
                                "allowed_values": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "filtering",
                        "name": "Filtering",
                        "description": (
                            "Filtering parameters define criteria for including or excluding annotations from "
                            "the dataset. Depending on the scenario, an appropriate filter configuration can speed up "
                            "the training process and/or improve the model performance by removing noisy annotations."
                        ),
                        "parameters": [
                            {
                                "type": "parameter_group",
                                "key": "min_annotation_pixels",
                                "name": "Minimum annotation pixels",
                                "description": "Minimum number of pixels in an annotation",
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable minimum annotation pixels filtering",
                                        "description": "Whether to apply minimum annotation pixels filtering",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Minimum annotation pixels",
                                        "description": "Minimum number of pixels in an annotation",
                                        "value": 20,
                                        "default_value": 1,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": 200000000,
                                        "allowed_values": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "min_annotation_objects",
                                "name": "Minimum annotation objects",
                                "description": "Minimum number of objects in an annotation",
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable minimum annotation objects filtering",
                                        "description": "Whether to apply minimum annotation objects filtering",
                                        "value": False,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Minimum annotation objects",
                                        "description": "Minimum number of objects in an annotation",
                                        "value": 1,
                                        "default_value": 1,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": None,
                                        "allowed_values": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "max_annotation_objects",
                                "name": "Maximum annotation objects",
                                "description": "Maximum number of objects in an annotation",
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable maximum annotation objects filtering",
                                        "description": "Whether to apply maximum annotation objects filtering",
                                        "value": False,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Maximum annotation objects",
                                        "description": "Maximum number of objects in an annotation",
                                        "value": 50,
                                        "default_value": 10000,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": None,
                                        "allowed_values": None,
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "augmentation",
                        "name": "Data augmentation",
                        "description": (
                            "Data augmentation is a technique used in machine learning to artificially expand a "
                            "training dataset by applying transformations (e.g., rotation, scaling, noise) to "
                            "existing data. It improves model generalization and reduces overfitting by "
                            "increasing data variability without collecting new samples."
                        ),
                        "parameters": [
                            {
                                "type": "parameter_group",
                                "key": "color_jitter",
                                "name": "Color jitter",
                                "description": (
                                    "Randomly adjust brightness, contrast, saturation, and hue of the image."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "brightness",
                                        "name": "Brightness range",
                                        "description": (
                                            "Range (min, max) of brightness adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "brightness. For example, (0.8, 1.2) means brightness can be reduced "
                                            "by 20% or increased by 20%."
                                        ),
                                        "value": [0.9, 1.1],
                                        "default_value": [0.8, 1.2],
                                        "value_type": "float_range",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "contrast",
                                        "name": "Contrast range",
                                        "description": (
                                            "Range (min, max) of contrast adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "contrast. For example, (0.5, 1.5) means contrast can be halved or "
                                            "increased by up to 50%."
                                        ),
                                        "value": [0.85, 1.15],
                                        "default_value": [0.75, 1.25],
                                        "value_type": "float_range",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "saturation",
                                        "name": "Saturation range",
                                        "description": (
                                            "Range (min, max) of saturation adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "saturation. For example, (0.5, 1.5) means saturation can be halved "
                                            "or increased by up to 50%."
                                        ),
                                        "value": [0.8, 1.2],
                                        "default_value": [0.9, 1.1],
                                        "value_type": "float_range",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "hue",
                                        "name": "Hue range",
                                        "description": (
                                            "Range (min, max) of hue adjustment values. "
                                            "A random value from this range will be added to the image hue. "
                                            "For example, (-0.05, 0.05) means hue can be shifted by up to ±0.05."
                                        ),
                                        "value": [-0.05, 0.05],
                                        "default_value": [-0.1, 0.1],
                                        "value_type": "float_range",
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying color jitter. "
                                            "A value of 0.5 means each image has a 50% chance to be color jittered."
                                        ),
                                        "value": 0.6,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                    },
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "type": "parameter_group",
                "key": "training",
                "name": "Training",
                "description": "Configurable parameters related to the learning phase (hyperparameters).",
                "parameters": [
                    {
                        "type": "parameter",
                        "key": "max_epochs",
                        "name": "Maximum epochs",
                        "description": (
                            "Maximum number of epochs to train the model. "
                            "An epoch is one complete pass through the training dataset."
                        ),
                        "value": 120,
                        "default_value": 250,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": None,
                        "allowed_values": None,
                    },
                    {
                        "type": "parameter_group",
                        "key": "early_stopping",
                        "name": "Early stopping",
                        "description": (
                            "Early stopping is a technique to prevent overfitting by stopping training "
                            "when performance on a validation set stops improving."
                        ),
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "enable",
                                "name": "Toggle early stopping",
                                "description": "Whether to stop training early when performance stops improving",
                                "value": True,
                                "default_value": False,
                                "value_type": "bool",
                                "allowed_values": None,
                            },
                            {
                                "type": "parameter",
                                "key": "patience",
                                "name": "Patience",
                                "description": (
                                    "Number of epochs with no improvement after which training will be stopped"
                                ),
                                "value": 5,
                                "default_value": 1,
                                "value_type": "int",
                                "min_value": 0,
                                "max_value": None,
                                "allowed_values": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter",
                        "key": "learning_rate",
                        "name": "Learning rate",
                        "description": (
                            "Learning rate for the optimizer, controlling the step size during model weight updates. "
                            "A smaller learning rate may lead to more stable convergence, while a larger learning rate "
                            "may speed up training but risk overshooting minima in the loss landscape."
                        ),
                        "value": 0.001,
                        "default_value": 0.0015,
                        "value_type": "float",
                        "min_value": 0,
                        "max_value": 1,
                        "allowed_values": None,
                    },
                    {
                        "type": "parameter",
                        "key": "input_size_width",
                        "name": "Input size width",
                        "description": (
                            "Width size in pixels for model input images. "
                            "Determines the horizontal resolution at which images are processed."
                        ),
                        "value": 256,
                        "default_value": 512,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": None,
                        "allowed_values": [128, 256, 512],
                    },
                    {
                        "type": "parameter",
                        "key": "input_size_height",
                        "name": "Input size height",
                        "description": (
                            "Height size in pixels for model input images. "
                            "Determines the vertical resolution at which images are processed."
                        ),
                        "value": 256,
                        "default_value": 512,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": None,
                        "allowed_values": [128, 256, 512],
                    },
                ],
            },
            {
                "key": "evaluation",
                "name": "Evaluation parameters",
                "description": "Configurable parameters related to the model evaluation.",
                "parameters": [
                    {
                        "type": "parameter",
                        "key": "validation_metric",
                        "name": "Validation metric",
                        "description": "Metric used to evaluate model performance during validation.",
                        "value": "default",
                        "default_value": "default",
                        "value_type": "str",
                        "allowed_values": ["default"],
                    },
                ],
                "type": "parameter_group",
            },
        ]
    }


class TestTrainingConfigurationView:
    def test_from_training_configuration(
        self, fxt_training_configuration, fxt_default_training_configuration, fxt_training_configuration_view_json
    ):
        view = TrainingConfigurationView.from_training_configuration(
            fxt_training_configuration, fxt_default_training_configuration
        )

        assert view.model_dump(mode="json") == fxt_training_configuration_view_json
