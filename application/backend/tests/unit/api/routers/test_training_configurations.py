# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from app.api.dependencies import get_training_configuration_service
from app.api.schemas import TrainingConfigurationView
from app.main import app
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
from app.services.model_manifest_service import ManifestNotFoundException
from app.services.training_configuration_service import TrainingConfigurationService


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
                        probability=0.5,
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
    """Create a default training configuration (as returned by get_default_by_model_architecture)."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(),
            training=AlgoLevelTrainingParameters(
                max_epochs=200,
                early_stopping=EarlyStopping(enable=False, patience=1),
                learning_rate=0.001,
                input_size_width=256,
                input_size_height=256,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_training_configuration_service() -> MagicMock:
    training_configuration_service = MagicMock(spec=TrainingConfigurationService)
    app.dependency_overrides[get_training_configuration_service] = lambda: training_configuration_service
    return training_configuration_service


class TestTrainingConfigurationEndpoints:
    def test_get_training_configuration_by_model_architecture(
        self,
        fxt_client,
        fxt_get_project,
        fxt_training_configuration_service,
        fxt_training_configuration,
        fxt_default_training_configuration,
    ):
        """Test successful retrieval of training configuration by model architecture."""
        project_id = fxt_get_project.id
        model_architecture_id = "object-detection-yolox-x"
        fxt_get_project.task_type = "classification"
        fxt_training_configuration_service.get_by_model_architecture.return_value = fxt_training_configuration

        mock_view = MagicMock(spec=TrainingConfigurationView)
        mock_view.model_dump.return_value = {"parameters": []}

        with (
            patch.object(TrainingConfigurationView, "from_training_configuration", return_value=mock_view) as mock_from,
            patch.object(
                TrainingConfigurationService,
                "get_default_by_model_architecture",
                return_value=fxt_default_training_configuration,
            ) as mock_get_default,
        ):
            response = fxt_client.get(
                f"/api/projects/{project_id}/training_configuration",
                params={"model_architecture_id": model_architecture_id},
            )

            fxt_training_configuration_service.get_by_model_architecture.assert_called_once_with(
                project_id=project_id, model_architecture_id=model_architecture_id
            )
            mock_get_default.assert_called_once_with(model_architecture_id=model_architecture_id)
            mock_from.assert_called_once_with(
                fxt_training_configuration, fxt_default_training_configuration, task_type="classification"
            )

        assert response.status_code == 200

    def test_get_training_configuration_by_model_architecture_manifest_not_found(
        self, fxt_client, fxt_get_project, fxt_training_configuration_service
    ):
        """Test retrieval of training configuration by model architecture when manifest is not found."""
        project_id = fxt_get_project.id
        model_architecture_id = "object-detection-yolox-x"
        fxt_training_configuration_service.get_by_model_architecture.side_effect = ManifestNotFoundException(
            model_architecture_id
        )

        response = fxt_client.get(
            f"/api/projects/{project_id}/training_configuration",
            params={"model_architecture_id": model_architecture_id},
        )

        assert response.status_code == 404

    def test_update_training_configuration_for_model_architecture(
        self,
        fxt_client,
        fxt_get_project,
        fxt_training_configuration,
        fxt_training_configuration_service,
        fxt_default_training_configuration,
    ):
        """Test successful update of training configuration for model architecture."""
        project_id = fxt_get_project.id
        model_architecture_id = "object-detection-yolox-x"
        fxt_get_project.task_type = "classification"
        fxt_training_configuration_service.get_by_model_architecture.return_value = fxt_training_configuration
        fxt_training_configuration_service.update.return_value = None
        fxt_get_project.task_type = "classification"

        update_payload = {
            "dataset_preparation.subset_split.training": 60,
            "dataset_preparation.subset_split.validation": 20,
            "dataset_preparation.subset_split.test": 20,
        }

        mock_view = MagicMock(spec=TrainingConfigurationView)
        mock_view.model_dump.return_value = {"parameters": []}

        with (
            patch.object(TrainingConfigurationView, "from_training_configuration", return_value=mock_view) as mock_from,
            patch.object(
                TrainingConfigurationService,
                "get_default_by_model_architecture",
                return_value=fxt_default_training_configuration,
            ),
        ):
            response = fxt_client.patch(
                f"/api/projects/{project_id}/training_configuration",
                params={"model_architecture_id": model_architecture_id},
                json=update_payload,
            )

            mock_from.assert_called_once()

        assert response.status_code == 200

        fxt_training_configuration_service.get_by_model_architecture.assert_called_once_with(
            project_id=project_id, model_architecture_id=model_architecture_id
        )
        assert fxt_training_configuration.task_level_parameters.dataset_preparation.subset_split.training == 60
        assert fxt_training_configuration.task_level_parameters.dataset_preparation.subset_split.validation == 20
        assert fxt_training_configuration.task_level_parameters.dataset_preparation.subset_split.test == 20
        fxt_training_configuration_service.update.assert_called_once_with(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            training_configuration=fxt_training_configuration,
        )

    def test_update_training_configuration_for_model_architecture_manifest_not_found(
        self, fxt_client, fxt_get_project, fxt_training_configuration_service
    ):
        """Test update of training configuration for model architecture when manifest is not found."""
        project_id = fxt_get_project.id
        model_architecture_id = "object-detection-yolox-x"
        fxt_training_configuration_service.get_by_model_architecture.side_effect = ManifestNotFoundException(
            model_architecture_id
        )
        fxt_training_configuration_service.update.return_value = None
        fxt_get_project.task_type = "classification"

        response = fxt_client.patch(
            f"/api/projects/{project_id}/training_configuration",
            params={"model_architecture_id": model_architecture_id},
            json={"training.max_epochs": 10},
        )

        assert response.status_code == 404
        fxt_training_configuration_service.update.assert_not_called()

    def test_update_training_configuration_for_model_architecture_invalid_update(
        self, fxt_client, fxt_get_project, fxt_training_configuration, fxt_training_configuration_service
    ):
        """Test update of training configuration for model architecture with an invalid update list."""
        project_id = fxt_get_project.id
        model_architecture_id = "object-detection-yolox-x"
        # Return a real TrainingConfiguration so apply_updates actually raises ValueError
        fxt_training_configuration_service.get_by_model_architecture.return_value = fxt_training_configuration
        fxt_training_configuration_service.update.return_value = None
        fxt_get_project.task_type = "classification"

        response = fxt_client.patch(
            f"/api/projects/{project_id}/training_configuration",
            params={"model_architecture_id": model_architecture_id},
            json={"nonexistent.parameter.path": 42},
        )

        assert response.status_code == 400
        fxt_training_configuration_service.update.assert_not_called()
