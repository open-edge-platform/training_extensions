# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.core.models import TaskType
from app.services.base_weights_service import BaseWeightsService
from app.services.training.base import PipelineContext
from app.services.training.models import TrainingParams
from app.services.training.steps.prepare_weights import MODEL_WEIGHTS_PATH, PrepareWeightsStep


@pytest.fixture
def fxt_base_weights_service() -> Mock:
    """Mock BaseWeightsService for testing."""
    return Mock(spec=BaseWeightsService)


@pytest.fixture
def fxt_prepare_weights_step(fxt_base_weights_service: Mock, tmp_path: Path) -> PrepareWeightsStep:
    """Create an OTXTrainer instance for testing."""
    return PrepareWeightsStep(fxt_base_weights_service, tmp_path)


class TestPrepareWeightsStep:
    """Test cases for the prepare weights step."""

    def test_prepare_weights_without_parent_model(
        self,
        fxt_prepare_weights_step: PrepareWeightsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
        fxt_base_weights_service: Mock,
        tmp_path: Path,
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

        # Act
        fxt_prepare_weights_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

        # Assert
        assert fxt_pipeline_ctx.get(MODEL_WEIGHTS_PATH) == expected_weights_path
        fxt_base_weights_service.get_local_weights_path.assert_called_once_with(
            task=TaskType.DETECTION, model_manifest_id="Object_Detection_YOLOX_S"
        )

    def test_prepare_weights_with_parent_model(
        self,
        fxt_prepare_weights_step: PrepareWeightsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
        fxt_base_weights_service: Mock,
        tmp_path: Path,
    ):
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
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )
        expected_weights_path.parent.mkdir(parents=True, exist_ok=True)
        expected_weights_path.touch()

        # Act
        fxt_prepare_weights_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

        # Assert
        assert fxt_pipeline_ctx.get(MODEL_WEIGHTS_PATH) == expected_weights_path

    def test_prepare_weights_with_parent_model_no_file_raises_error(
        self,
        fxt_prepare_weights_step: PrepareWeightsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
        fxt_base_weights_service: Mock,
        tmp_path: Path,
    ):
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
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )

        # Act
        with pytest.raises(FileNotFoundError, match=f"Parent model weights not found at {expected_weights_path}"):
            fxt_prepare_weights_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

    def test_prepare_weights_with_parent_model_no_project_id_raises_error(
        self,
        fxt_prepare_weights_step: PrepareWeightsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
        fxt_base_weights_service: Mock,
        tmp_path: Path,
    ):
        """Test that ValueError is raised when parent model revision ID is provided without project ID."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=uuid4(),
            project_id=None,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for parent model weights preparation"):
            fxt_prepare_weights_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

    def test_get_name(self, fxt_prepare_weights_step: PrepareWeightsStep):
        """Test that the step returns the correct name."""
        assert fxt_prepare_weights_step.get_name() == "Prepare Model Weights"
