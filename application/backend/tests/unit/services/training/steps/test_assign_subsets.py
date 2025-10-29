# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.models import DatasetItemSubset, TaskType
from app.services.training.base import PipelineContext
from app.services.training.models import TrainingParams
from app.services.training.steps.assign_subsets import AssignSubsetsStep
from app.services.training.steps.subset_assignment import (
    DatasetItemWithLabels,
    SubsetAssigner,
    SubsetAssignment,
    SubsetDistribution,
    SubsetService,
)


@pytest.fixture
def fxt_subset_service() -> Mock:
    """Mock SubsetService for testing."""
    return Mock(spec=SubsetService)


@pytest.fixture
def fxt_assigner() -> Mock:
    """Mock SubsetAssigner for testing."""
    return Mock(spec=SubsetAssigner)


@pytest.fixture
def fxt_assign_subsets_step(
    fxt_subset_service: Mock,
    fxt_assigner: Mock,
    fxt_db_session_factory,
) -> AssignSubsetsStep:
    """Create an AssignSubsetsStep instance for testing."""
    return AssignSubsetsStep(fxt_subset_service, fxt_assigner, fxt_db_session_factory)


class TestAssignSubsetsStep:
    """Test cases for the assign subsets step."""

    def test_assign_subsets_with_unassigned_items(
        self,
        fxt_assign_subsets_step: AssignSubsetsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
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
        fxt_assign_subsets_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id, fxt_db_session)
        fxt_subset_service.get_subset_distribution.assert_called_once_with(project_id, fxt_db_session)
        fxt_assigner.assign.assert_called_once()
        fxt_subset_service.update_subset_assignments.assert_called_once_with(
            project_id, expected_assignments, fxt_db_session
        )

        # Verify progress reporting
        assert fxt_execution_ctx.report_progress.call_count >= 4

    def test_assign_subsets_with_no_unassigned_items(
        self,
        fxt_assign_subsets_step: AssignSubsetsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
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

        # Act
        fxt_assign_subsets_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id, fxt_db_session)
        fxt_subset_service.get_subset_distribution.assert_not_called()
        fxt_assigner.assign.assert_not_called()
        fxt_subset_service.update_subset_assignments.assert_not_called()

        # Verify progress reporting for no items
        fxt_execution_ctx.report_progress.assert_any_call("No unassigned items found")

    def test_assign_subsets_without_project_id_raises_error(
        self,
        fxt_assign_subsets_step: AssignSubsetsStep,
        fxt_execution_ctx: Mock,
        fxt_pipeline_ctx: PipelineContext,
    ):
        """Test that ValueError is raised when no project ID is provided."""
        # Arrange
        training_params = TrainingParams(
            project_id=None,
            model_architecture_id="Object_Detection_YOLOX_S",
            task_type=TaskType.DETECTION,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for subset assignment"):
            fxt_assign_subsets_step.execute(fxt_execution_ctx, training_params, fxt_pipeline_ctx)

    def test_get_name(self, fxt_assign_subsets_step: AssignSubsetsStep):
        """Test that the step returns the correct name."""
        assert fxt_assign_subsets_step.get_name() == "Assign Subsets"
