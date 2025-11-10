# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.api.dependencies import get_training_configuration_service
from app.main import app
from app.models.training_configuration.configuration import (
    GlobalDatasetPreparationParameters,
    GlobalParameters,
    SubsetSplit,
    TrainingConfiguration,
)
from app.models.training_configuration.hyperparameters import Hyperparameters
from app.services.training_configuration_service import TrainingConfigurationService


@pytest.fixture
def fxt_training_configuration() -> TrainingConfiguration:
    """Create a mock training configuration."""
    return TrainingConfiguration(
        model_manifest_id="Custom_Image_Classification_EfficientNet-B0",
        global_parameters=GlobalParameters(
            dataset_preparation=GlobalDatasetPreparationParameters(
                subset_split=SubsetSplit(),
            )
        ),
        hyperparameters=Hyperparameters(),
    )


@pytest.fixture
def fxt_training_configuration_service() -> MagicMock:
    training_configuration_service = MagicMock(spec=TrainingConfigurationService)
    app.dependency_overrides[get_training_configuration_service] = lambda: training_configuration_service
    return training_configuration_service


class TestTrainingConfigurationEndpoints:
    def test_get_training_configuration_success(
        self, fxt_client, fxt_training_configuration, fxt_training_configuration_service
    ):
        """Test successful retrieval of training configuration."""
        project_id = uuid4()
        fxt_training_configuration_service.get_training_configuration.return_value = fxt_training_configuration

        response = fxt_client.get(f"/api/projects/{project_id}/training_configuration")

        assert response.status_code == 200
        data = response.json()
        assert "dataset_preparation" in data
        assert "training" in data
        assert "evaluation" in data
