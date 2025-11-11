# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.models.training_configuration import (
    AugmentationParameters,
    DatasetPreparationParameters,
    EarlyStopping,
    EvaluationParameters,
    Hyperparameters,
    RandomResizeCrop,
    TrainingHyperParameters,
)
from app.models.training_configuration.configuration import (
    Filtering,
    GlobalDatasetPreparationParameters,
    GlobalParameters,
    MaxAnnotationObjects,
    MaxAnnotationPixels,
    MinAnnotationObjects,
    MinAnnotationPixels,
    PartialTrainingConfiguration,
    SubsetSplit,
    TrainingConfiguration,
)
from app.services.tools import ConfigurationOverlayTools


@pytest.fixture
def ftx_hyperparameters():
    yield Hyperparameters(
        dataset_preparation=DatasetPreparationParameters(
            augmentation=AugmentationParameters(
                random_resize_crop=RandomResizeCrop(enable=True, crop_ratio_range=[0.1, 0.8]),
            )
        ),
        training=TrainingHyperParameters(
            max_epochs=100,
            early_stopping=EarlyStopping(enable=True, patience=10),
            learning_rate=0.001,
        ),
        evaluation=EvaluationParameters(),
    )


@pytest.fixture
def fxt_global_parameters():
    yield GlobalParameters(
        dataset_preparation=GlobalDatasetPreparationParameters(
            subset_split=SubsetSplit(
                training=70,
                validation=20,
                test=10,
                auto_selection=True,
                remixing=False,
            ),
            filtering=Filtering(
                min_annotation_pixels=MinAnnotationPixels(enable=True, min_annotation_pixels=10),
                max_annotation_pixels=MaxAnnotationPixels(enable=True, max_annotation_pixels=1000),
                min_annotation_objects=MinAnnotationObjects(enable=True, min_annotation_objects=5),
                max_annotation_objects=MaxAnnotationObjects(enable=True, max_annotation_objects=100),
            ),
        )
    )


@pytest.fixture
def fxt_training_configuration_task_level(fxt_global_parameters, ftx_hyperparameters):
    yield TrainingConfiguration(
        global_parameters=fxt_global_parameters,
        hyperparameters=ftx_hyperparameters,
    )


class TestConfigurationService:
    @pytest.mark.parametrize(
        "d, expected",
        [
            # Empty dict
            ({}, {}),
            # Dict with no None values
            ({"a": 1, "b": "test"}, {"a": 1, "b": "test"}),
            # Dict with only None Values
            ({"a": 1, "b": {"b1": None}}, {"a": 1}),
            ({"a": 1, "b": [{"b1": None}]}, {"a": 1}),
            # Dict with None values
            ({"a": 1, "b": None, "c": "test"}, {"a": 1, "c": "test"}),
            # Dict with nested dict containing None values
            ({"a": 1, "b": {"x": None, "y": 2}}, {"a": 1, "b": {"y": 2}}),
            # Dict with list containing dicts with None values
            ({"a": 1, "b": [{"x": None, "y": 2}, {"z": 3}]}, {"a": 1, "b": [{"y": 2}, {"z": 3}]}),
            # Complex nested scenario
            (
                {"a": 1, "b": None, "c": {"d": None, "e": {"f": None, "g": 2}, "h": [{"i": None, "j": 3}]}},
                {"a": 1, "c": {"e": {"g": 2}, "h": [{"j": 3}]}},
            ),
        ],
    )
    def test_delete_none_from_dict(self, d, expected) -> None:
        result = ConfigurationOverlayTools.delete_none_from_dict(d)
        assert result == expected
        # Ensure the function modifies the dict in-place
        assert result is d

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            # Empty dicts
            ({}, {}, {}),
            # Empty target dict
            ({}, {"x": 1}, {"x": 1}),
            # Empty source dict
            ({"x": 1}, {}, {"x": 1}),
            # Non-overlapping keys
            ({"x": 1}, {"y": 2}, {"x": 1, "y": 2}),
            # Overlapping keys (non-dict values)
            ({"x": 1}, {"x": 2}, {"x": 2}),
            # Simple nested dict
            ({"x": 1, "y": {"a": 2}}, {"y": {"b": 3}, "z": 4}, {"x": 1, "y": {"a": 2, "b": 3}, "z": 4}),
            # Complex nested scenario
            (
                {"a": 1, "b": {"c": 2, "d": {"e": 3}}},
                {"b": {"d": {"f": 4}, "g": 5}},
                {"a": 1, "b": {"c": 2, "d": {"e": 3, "f": 4}, "g": 5}},
            ),
            # Overwrite dict with non-dict
            ({"a": {"b": 1}}, {"a": 2}, {"a": 2}),
            # Overwrite non-dict with dict
            ({"a": 1}, {"a": {"b": 2}}, {"a": {"b": 2}}),
        ],
    )
    def test_merge_deep_dict(self, a, b, expected) -> None:
        result = ConfigurationOverlayTools.merge_deep_dict(a, b)
        assert result == expected
        # Ensure the function modifies the first dict in-place
        assert result is a

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            # Empty dicts
            ({}, {}, {}),
            # Empty target dict - nothing should be merged
            ({}, {"x": 1}, {}),
            # Empty source dict
            ({"x": 1}, {}, {"x": 1}),
            # Non-overlapping keys - nothing should be merged
            ({"x": 1}, {"y": 2}, {"x": 1}),
            # Overlapping keys (non-dict values)
            ({"x": 1}, {"x": 2}, {"x": 2}),
            # Simple nested dict - only common keys updated
            ({"x": 1, "y": {"a": 2}}, {"y": {"a": 3, "b": 4}, "z": 5}, {"x": 1, "y": {"a": 3}}),
            # Complex nested scenario - only common keys updated
            (
                {"a": 1, "b": {"c": 2, "d": {"e": 3}}},
                {"b": {"c": 5, "d": {"e": 6, "f": 7}, "g": 8}, "h": 9},
                {"a": 1, "b": {"c": 5, "d": {"e": 6}}},
            ),
            # Overwrite dict with non-dict
            ({"a": {"b": 1}}, {"a": 2}, {"a": 2}),
            # Overwrite non-dict with dict
            ({"a": 1}, {"a": {"b": 2}}, {"a": {"b": 2}}),
        ],
    )
    def test_merge_deep_dict_common_parameters_only(self, a, b, expected) -> None:
        result = ConfigurationOverlayTools.merge_deep_dict(a, b, common_parameters_only=True)
        assert result == expected
        # Ensure the function modifies the first dict in-place
        assert result is a

    def test_overlay_configurations(self, fxt_training_configuration_task_level) -> None:
        # Arrange
        # Create base configuration
        base_parameters = {
            "global_parameters": {
                "dataset_preparation": {
                    "subset_split": {"training": 70, "validation": 20, "test": 10, "auto_selection": True},
                    "filtering": {"min_annotation_pixels": {"enable": False, "min_annotation_pixels": 1}},
                }
            },
        }
        base_partial_config = PartialTrainingConfiguration(**base_parameters)  # type: ignore[arg-type]

        # Create overlay configuration with some changes
        overlay_parameters_1 = {
            "global_parameters": {
                "dataset_preparation": {
                    "subset_split": {"training": 60, "validation": 30, "test": 10, "remixing": True},
                    "filtering": {"max_annotation_pixels": {"enable": True, "max_annotation_pixels": 5000}},
                }
            },
            "hyperparameters": {
                "training": {
                    "max_epochs": 32,
                    "learning_rate": 0.01,
                }
            },
        }
        overlay_parameters_2 = {"hyperparameters": {"training": {"learning_rate": 0.05}}}
        overlay_config_1 = PartialTrainingConfiguration(**overlay_parameters_1)  # type: ignore[arg-type]
        overlay_config_2 = PartialTrainingConfiguration(**overlay_parameters_2)  # type: ignore[arg-type]

        expected_parameters = {
            "global_parameters": {
                "dataset_preparation": {
                    "subset_split": {
                        "training": 60,
                        "validation": 30,
                        "test": 10,
                        "remixing": True,
                        "auto_selection": True,
                    },
                    "filtering": {
                        "min_annotation_pixels": {"enable": False, "min_annotation_pixels": 1},
                        "max_annotation_pixels": {"enable": True, "max_annotation_pixels": 5000},
                    },
                }
            },
            "hyperparameters": {
                "training": {
                    "max_epochs": 32,
                    "learning_rate": 0.05,  # This should be the last value applied
                }
            },
        }
        expected_partial_overlay_config = PartialTrainingConfiguration(**expected_parameters)  # type: ignore[arg-type]

        # Act
        full_config_overlay = ConfigurationOverlayTools.overlay_training_configurations(
            fxt_training_configuration_task_level, base_partial_config, overlay_config_1, overlay_config_2
        )
        partial_overlay = ConfigurationOverlayTools.overlay_training_configurations(
            base_partial_config, overlay_config_1, overlay_config_2, validate_full_config=False
        )

        # Assert
        full_config_dataset_preparation = full_config_overlay.global_parameters.dataset_preparation
        assert partial_overlay.model_dump() == expected_partial_overlay_config.model_dump()
        assert full_config_dataset_preparation.subset_split.training == 60
        assert full_config_dataset_preparation.subset_split.validation == 30
        assert full_config_dataset_preparation.subset_split.remixing
        assert full_config_dataset_preparation.filtering.max_annotation_pixels
        assert full_config_dataset_preparation.filtering.max_annotation_pixels.enable
        assert full_config_dataset_preparation.filtering.min_annotation_pixels
        assert not full_config_dataset_preparation.filtering.min_annotation_pixels.enable
        assert full_config_dataset_preparation.filtering.min_annotation_pixels.value == 1
        assert full_config_overlay.hyperparameters.training
        assert full_config_overlay.hyperparameters.training.max_epochs == 32
        assert full_config_overlay.hyperparameters.training.learning_rate == 0.05

    def test_overlay_configurations_common_hyperparameters_only(self) -> None:
        # Arrange
        # Create base configuration
        base_parameters = {
            "hyperparameters": {
                "training": {
                    "allowed_values_input_size": [224, 256],
                    "input_size_width": 224,
                    "input_size_height": 224,
                }
            }
        }
        base_partial_config = PartialTrainingConfiguration(**base_parameters)  # type: ignore[arg-type]

        # Create overlay configuration with some changes
        overlay_parameters_1 = {
            "global_parameters": {
                "dataset_preparation": {
                    "subset_split": {"training": 60, "validation": 30, "test": 10, "remixing": True},
                    "filtering": {"max_annotation_pixels": {"enable": True, "max_annotation_pixels": 5000}},
                }
            },
            "hyperparameters": {
                "training": {
                    "input_size_width": 256,
                    "max_epochs": 32,
                }
            },
        }
        overlay_parameters_2 = {"hyperparameters": {"training": {"learning_rate": 0.05}}}
        overlay_config_1 = PartialTrainingConfiguration(**overlay_parameters_1)  # type: ignore[arg-type]
        overlay_config_2 = PartialTrainingConfiguration(**overlay_parameters_2)  # type: ignore[arg-type]

        expected_parameters = {
            "global_parameters": {
                "dataset_preparation": {
                    "subset_split": {"training": 60, "validation": 30, "test": 10, "remixing": True},
                    "filtering": {"max_annotation_pixels": {"enable": True, "max_annotation_pixels": 5000}},
                }
            },
            "hyperparameters": {
                "training": {
                    "allowed_values_input_size": [224, 256],
                    "input_size_width": 256,
                    "input_size_height": 224,
                }
            },
        }
        expected_partial_overlay_config = TrainingConfiguration(**expected_parameters)  # type: ignore[arg-type]

        # Act
        full_config_overlay = ConfigurationOverlayTools.overlay_training_configurations(
            base_partial_config,
            overlay_config_1,
            overlay_config_2,
            common_hyperparameters_only=True,
        )

        # Assert
        assert full_config_overlay.model_dump() == expected_partial_overlay_config.model_dump()
