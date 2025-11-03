# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.core.run import ExecutionContext
from app.models import DatasetItemSubset, TaskType
from app.services.base_weights_service import BaseWeightsService
from app.services.training.models import TrainingParams
from app.services.training.otx_trainer import OTXTrainer
from app.services.training.subset_assignment import (
    DatasetItemWithLabels,
    SubsetAssigner,
    SubsetAssignment,
    SubsetDistribution,
    SubsetService,
)


@pytest.fixture
def fxt_weights_service() -> Mock:
    """Create a mock BaseWeightsService."""
    return Mock(spec=BaseWeightsService)


@pytest.fixture
def fxt_subset_service() -> Mock:
    """Mock SubsetService for testing."""
    return Mock(spec=SubsetService)


@pytest.fixture
def fxt_assigner() -> Mock:
    """Mock SubsetAssigner for testing."""
    return Mock(spec=SubsetAssigner)


@pytest.fixture
def fxt_otx_trainer(
    tmp_path: Path,
    fxt_weights_service: Mock,
    fxt_subset_service: Mock,
    fxt_assigner: Mock,
    fxt_db_session_factory: Callable,
) -> Callable[[TrainingParams], OTXTrainer]:
    """Create an OTXTrainer instance."""

    def create_otx_trainer(params: TrainingParams) -> OTXTrainer:
        otx_trainer = OTXTrainer(
            data_dir=tmp_path,
            base_weights_service=fxt_weights_service,
            subset_service=fxt_subset_service,
            subset_assigner=fxt_assigner,
            db_session_factory=fxt_db_session_factory,
        )
        execution_ctx = Mock(spec=ExecutionContext)
        execution_ctx.report = Mock()
        execution_ctx.heartbeat = Mock()
        otx_trainer._ctx = execution_ctx
        otx_trainer._training_params = params
        return otx_trainer

    return create_otx_trainer


class TestOTXTrainerPrepareWeights:
    """Tests for the OTXTrainer.prepare_weights method."""

    def test_prepare_weights_without_parent_model(
        self,
        fxt_weights_service: Mock,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
    ):
        """Test preparing weights when no parent model revision ID is provided."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=None,
        )
        otx_trainer = fxt_otx_trainer(training_params)

        expected_weights_path = Path("/path/to/weights.pth")
        fxt_weights_service.get_local_weights_path.return_value = expected_weights_path

        # Act
        weights_path = otx_trainer.prepare_weights()

        # Assert
        assert weights_path == expected_weights_path
        fxt_weights_service.get_local_weights_path.assert_called_once_with(
            task=TaskType.DETECTION, model_manifest_id="Object_Detection_YOLOX_S"
        )

    def test_prepare_weights_with_parent_model(
        self,
        tmp_path: Path,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
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
        otx_trainer = fxt_otx_trainer(training_params)

        # Act
        weights_path = otx_trainer.prepare_weights()

        # Assert
        assert weights_path == expected_weights_path

    def test_prepare_weights_with_parent_model_no_file_raises_error(
        self,
        tmp_path: Path,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
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
        otx_trainer = fxt_otx_trainer(training_params)

        # Act
        with pytest.raises(FileNotFoundError, match=f"Parent model weights not found at {expected_weights_path}"):
            otx_trainer.prepare_weights()

    def test_prepare_weights_with_parent_model_no_project_id_raises_error(
        self,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
    ):
        """Test that ValueError is raised when parent model revision ID is provided without project ID."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
            parent_model_revision_id=uuid4(),
            project_id=None,
        )
        otx_trainer = fxt_otx_trainer(training_params)

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for parent model weights preparation"):
            otx_trainer.prepare_weights()


class TestOTXTrainerAssignSubsets:
    """Tests for the OTXTrainer.assign_subsets method."""

    def test_assign_subsets_with_unassigned_items(
        self,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
        fxt_db_session: Mock,
    ):
        """Test assigning subsets when unassigned items exist."""
        # Arrange
        project_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
        )
        otx_trainer = fxt_otx_trainer(training_params)

        # Create mock unassigned items
        unassigned_items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={uuid4(), uuid4()}),
            DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}),
            DatasetItemWithLabels(item_id=uuid4(), labels={uuid4(), uuid4(), uuid4()}),
        ]
        fxt_subset_service.get_unassigned_items_with_labels.return_value = unassigned_items

        # Mock current distribution
        current_distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 10,
                DatasetItemSubset.VALIDATION: 3,
                DatasetItemSubset.TESTING: 2,
            }
        )
        fxt_subset_service.get_subset_distribution.return_value = current_distribution

        # Mock assignments
        expected_assignments = [
            SubsetAssignment(item_id=unassigned_items[0].item_id, subset=DatasetItemSubset.TRAINING),
            SubsetAssignment(item_id=unassigned_items[1].item_id, subset=DatasetItemSubset.VALIDATION),
            SubsetAssignment(item_id=unassigned_items[2].item_id, subset=DatasetItemSubset.TESTING),
        ]
        fxt_assigner.assign.return_value = expected_assignments

        # Act
        otx_trainer.assign_subsets()

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id, fxt_db_session)
        fxt_subset_service.get_subset_distribution.assert_called_once_with(project_id, fxt_db_session)
        fxt_assigner.assign.assert_called_once()
        fxt_subset_service.update_subset_assignments.assert_called_once_with(
            project_id, expected_assignments, fxt_db_session
        )

    def test_assign_subsets_with_no_unassigned_items(
        self,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
        fxt_db_session: Mock,
    ):
        """Test assigning subsets when no unassigned items exist."""
        # Arrange
        project_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
        )
        fxt_subset_service.get_unassigned_items_with_labels.return_value = []
        otx_trainer = fxt_otx_trainer(training_params)

        # Act
        otx_trainer.assign_subsets()

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id, fxt_db_session)
        fxt_subset_service.get_subset_distribution.assert_not_called()
        fxt_assigner.assign.assert_not_called()
        fxt_subset_service.update_subset_assignments.assert_not_called()

    def test_assign_subsets_without_project_id_raises_error(
        self,
        fxt_otx_trainer: Callable[[TrainingParams], OTXTrainer],
    ):
        """Test that ValueError is raised when no project ID is provided."""
        # Arrange
        training_params = TrainingParams(
            project_id=None,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
        )
        otx_trainer = fxt_otx_trainer(training_params)

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for subset assignment"):
            otx_trainer.assign_subsets()
