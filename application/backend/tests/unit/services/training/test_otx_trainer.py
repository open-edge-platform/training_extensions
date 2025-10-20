# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, call
from uuid import uuid4

import pytest

from app.core.models import TaskType
from app.services.base_weights_service import BaseWeightsService
from app.services.training.models import TrainingParams
from app.services.training.otx import OTXTrainer


@pytest.fixture
def fxt_base_weights_service() -> Mock:
    """Mock BaseWeightsService for testing."""
    return Mock(spec=BaseWeightsService)


@pytest.fixture
def fxt_otx_trainer(fxt_base_weights_service: Mock) -> OTXTrainer:
    """Create an OTXTrainer instance for testing."""
    return OTXTrainer(fxt_base_weights_service)


class TestOTXTrainerPrepareWeights:
    """Test cases for the _prepare_weights method."""

    def test_prepare_weights_without_parent_model(
        self, fxt_otx_trainer: OTXTrainer, fxt_base_weights_service: Mock, tmp_path: Path
    ):
        """Test preparing weights when no parent model revision ID is provided."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=None,
        )
        expected_weights_path = tmp_path / "model.pth"
        fxt_base_weights_service.get_local_weights_path.return_value = expected_weights_path
        report_fn = Mock()

        # Act
        result = fxt_otx_trainer._prepare_weights(tmp_path, training_params, report_fn)

        # Assert
        assert result == expected_weights_path
        fxt_base_weights_service.get_local_weights_path.assert_called_once_with(
            task=TaskType.DETECTION, model_manifest_id="Object_Detection_YOLOX_S"
        )
        report_fn.assert_has_calls([call("Preparing weights for training"), call("Base weights preparation completed")])

    def test_prepare_weights_with_parent_model(self, fxt_otx_trainer: OTXTrainer, tmp_path: Path):
        """Test preparing weights when parent model revision ID is provided."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=parent_model_revision_id,
        )
        report_fn = Mock()
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )
        expected_weights_path.parent.mkdir(parents=True, exist_ok=True)
        expected_weights_path.touch()

        # Act
        result = fxt_otx_trainer._prepare_weights(tmp_path, training_params, report_fn)

        # Assert
        assert result == expected_weights_path
        report_fn.assert_has_calls(
            [call("Preparing weights for training"), call("Parent model weights preparation completed")]
        )

    def test_prepare_weights_with_parent_model_no_file_raises_error(self, fxt_otx_trainer: OTXTrainer, tmp_path: Path):
        """Test that FileNotFoundError is raised when parent model weights file is missing."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=parent_model_revision_id,
        )
        report_fn = Mock()
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )

        # Act
        with pytest.raises(FileNotFoundError, match=f"Parent model weights not found at {expected_weights_path}"):
            fxt_otx_trainer._prepare_weights(tmp_path, training_params, report_fn)

        # Assert
        report_fn.assert_called_once_with("Preparing weights for training")

    def test_prepare_weights_with_parent_model_no_project_id_raises_error(
        self, fxt_otx_trainer: OTXTrainer, tmp_path: Path
    ):
        """Test that ValueError is raised when parent model revision ID is provided without project ID."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=uuid4(),
            project_id=None,
        )
        report_fn = Mock()

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for parent model weights preparation"):
            fxt_otx_trainer._prepare_weights(tmp_path, training_params, report_fn)
