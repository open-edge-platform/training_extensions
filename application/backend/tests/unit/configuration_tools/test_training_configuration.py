# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

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
    PartialGlobalParameters,
    PartialTrainingConfiguration,
    SubsetSplit,
    TrainingConfiguration,
)


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
            input_size_width=32,
            input_size_height=32,
            allowed_values_input_size=[32, 64, 128],
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
                min_annotation_pixels=MinAnnotationPixels(enable=True, value=10),
                max_annotation_pixels=MaxAnnotationPixels(enable=True, value=1000),
                min_annotation_objects=MinAnnotationObjects(enable=True, value=5),
                max_annotation_objects=MaxAnnotationObjects(enable=True, value=100),
            ),
        )
    )


class TestTrainingConfiguration:
    def test_valid_training_configuration(self, fxt_global_parameters, ftx_hyperparameters) -> None:
        # Create a valid TrainingConfiguration
        training_config = TrainingConfiguration(
            global_parameters=fxt_global_parameters,
            hyperparameters=ftx_hyperparameters,
        )

        # Verify the created config properties
        assert training_config.global_parameters.dataset_preparation.subset_split.training == 70
        assert training_config.global_parameters.dataset_preparation.subset_split.validation == 20
        assert training_config.global_parameters.dataset_preparation.subset_split.test == 10
        assert training_config.hyperparameters.training is not None
        assert training_config.hyperparameters.training.max_epochs == 100
        assert training_config.hyperparameters.training.early_stopping
        assert training_config.hyperparameters.training.early_stopping.enable
        assert training_config.hyperparameters.training.early_stopping.patience == 10

    def test_invalid_subset_split(self, ftx_hyperparameters) -> None:
        # Test that validation fails when subset percentages don't add up to 100
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfiguration(
                global_parameters=GlobalParameters(
                    dataset_preparation=GlobalDatasetPreparationParameters(
                        subset_split=SubsetSplit(
                            training=60,  # Sum is 90, not 100
                            validation=20,
                            test=10,
                            auto_selection=True,
                            remixing=False,
                        ),
                        filtering=Filtering(
                            min_annotation_pixels=MinAnnotationPixels(),
                            max_annotation_pixels=MaxAnnotationPixels(),
                            min_annotation_objects=MinAnnotationObjects(),
                            max_annotation_objects=MaxAnnotationObjects(),
                        ),
                    )
                ),
                hyperparameters=ftx_hyperparameters,
            )

        assert "Sum of subsets should be equal to 100" in str(excinfo.value)

    def test_invalid_annotation_pixels(self, ftx_hyperparameters) -> None:
        # Test validation for annotation pixels
        with pytest.raises(ValidationError):
            TrainingConfiguration(
                global_parameters=GlobalParameters(
                    dataset_preparation=GlobalDatasetPreparationParameters(
                        subset_split=SubsetSplit(),
                        filtering=Filtering(
                            min_annotation_pixels=MinAnnotationPixels(
                                enable=True,
                                value=0,  # Invalid: must be > 0  # pyrefly: ignore[bad-argument-type]
                            ),
                            max_annotation_pixels=MaxAnnotationPixels(),
                            min_annotation_objects=MinAnnotationObjects(),
                            max_annotation_objects=MaxAnnotationObjects(),
                        ),
                    )
                ),
                hyperparameters=ftx_hyperparameters,
            )

    def test_partial_training_configuration(self, fxt_global_parameters, ftx_hyperparameters) -> None:
        """Test that PartialTrainingConfiguration works correctly with both partial and complete configurations."""
        partial_training_config_incomplete = PartialTrainingConfiguration.model_validate(
            {
                "model_manifest_id": "test_manifest",
                "global_parameters": {
                    "dataset_preparation": {
                        "filtering": {
                            "min_annotation_pixels": {
                                "value": 42,
                            },
                        }
                    }
                },
            }
        )
        global_parameters = partial_training_config_incomplete.global_parameters
        assert partial_training_config_incomplete.model_manifest_id == "test_manifest"
        assert global_parameters.dataset_preparation.filtering.min_annotation_pixels
        assert global_parameters.dataset_preparation.filtering.min_annotation_pixels.value == 42
        assert global_parameters.dataset_preparation.filtering.min_annotation_pixels.enable is None
        assert global_parameters.dataset_preparation.subset_split is None

        # Full configuration
        training_config = TrainingConfiguration(
            global_parameters=fxt_global_parameters,
            hyperparameters=ftx_hyperparameters,
        )
        full_training_config_dict = training_config.model_dump()
        partial_training_config_full = PartialTrainingConfiguration.model_validate(full_training_config_dict)

        assert partial_training_config_full.model_dump() == training_config.model_dump()

    def test_partial_global_parameters(self, fxt_global_parameters) -> None:
        """Test that PartialGlobalParameters works correctly with both partial and complete configurations."""
        # Test with a partial configuration
        partial_global_params = PartialGlobalParameters.model_validate(
            {
                "dataset_preparation": {
                    "subset_split": {
                        "training": 42,
                        "validation": 48,
                        "test": 10,
                    }
                }
            }
        )

        # Verify that specified fields are set correctly
        assert partial_global_params.dataset_preparation.subset_split.training == 42
        assert partial_global_params.dataset_preparation.subset_split.validation == 48
        assert partial_global_params.dataset_preparation.subset_split.test == 10

        # Verify that subset validator still works e.g. sum must be 100
        with pytest.raises(ValidationError):
            PartialGlobalParameters.model_validate(
                {
                    "dataset_preparation": {
                        "subset_split": {
                            "training": 1,
                            "validation": 1,
                            "test": 1,
                        }
                    }
                }
            )

        # Verify that unspecified fields are None
        assert partial_global_params.dataset_preparation.subset_split
        assert partial_global_params.dataset_preparation.filtering is None

        # Test with a full configuration
        full_global_params_dict = fxt_global_parameters.model_dump()
        partial_global_params_full = PartialGlobalParameters.model_validate(full_global_params_dict)

        assert partial_global_params_full.model_dump() == fxt_global_parameters.model_dump()

    def test_validate_subsets(self) -> None:
        # subsets dont add app to 100
        with pytest.raises(ValueError):
            PartialGlobalParameters.model_validate(
                {
                    "dataset_preparation": {
                        "subset_split": {
                            "training": 1,
                            "validation": 1,
                            "test": 1,
                        }
                    }
                }
            )

        # test doesn't have items
        with pytest.raises(ValueError):
            PartialGlobalParameters.model_validate(
                {
                    "dataset_preparation": {
                        "subset_split": {
                            "training": 50,
                            "validation": 49,
                            "test": 1,
                            "dataset_size": 2,
                        }
                    }
                }
            )
