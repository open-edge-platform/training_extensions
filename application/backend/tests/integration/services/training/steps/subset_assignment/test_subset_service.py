# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, LabelDB, ProjectDB
from app.models import DatasetItemSubset
from app.services.training.subset_assignment import SubsetAssignment, SubsetService
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture(autouse=True)
def setup_project_with_dataset_items(
    fxt_db_projects: list[ProjectDB],
    fxt_db_labels: list[LabelDB],
    fxt_default_distribution: dict[DatasetItemSubset, int],
    db_session: Session,
) -> None:
    """Fixture to set up a project with dataset items in the database."""

    db_label = fxt_db_labels[0]
    (
        ProjectTestDataFactory(db_session)
        .with_project(fxt_db_projects[0])
        .with_label(db_label)
        .with_dataset_items(fxt_default_distribution)
        .with_item_labels(db_label)
        .build()
    )


@pytest.fixture
def fxt_subset_service(db_session: Session) -> SubsetService:
    """Fixture to create a SubsetService instance."""
    return SubsetService(db_session)


@pytest.fixture
def fxt_default_distribution() -> dict[DatasetItemSubset, int]:
    """Fixture to provide a default subset distribution."""
    return {
        DatasetItemSubset.TRAINING: 70,
        DatasetItemSubset.VALIDATION: 20,
        DatasetItemSubset.TESTING: 10,
        DatasetItemSubset.UNASSIGNED: 50,
    }


class TestSubsetServiceIntegration:
    """Integration tests for SubsetService."""

    def test_get_unassigned_items_with_labels(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
    ) -> None:
        """Test retrieving unassigned dataset items with labels."""
        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id)
        assert len(items) == fxt_default_distribution.get(DatasetItemSubset.UNASSIGNED)  # based on default distribution

    def test_get_subset_distribution(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
    ):
        """Test retrieving subset distribution."""
        distribution = fxt_subset_service.get_subset_distribution(fxt_project_id)
        assert distribution.get_count(DatasetItemSubset.TRAINING) == fxt_default_distribution.get(
            DatasetItemSubset.TRAINING
        )
        assert distribution.get_count(DatasetItemSubset.VALIDATION) == fxt_default_distribution.get(
            DatasetItemSubset.VALIDATION
        )
        assert distribution.get_count(DatasetItemSubset.TESTING) == fxt_default_distribution.get(
            DatasetItemSubset.TESTING
        )
        assert distribution.get_count(DatasetItemSubset.UNASSIGNED) == fxt_default_distribution.get(
            DatasetItemSubset.UNASSIGNED
        )

    def test_update_subset_assignments(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
        db_session: Session,
    ):
        """Test updating subset assignments."""
        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id)
        assignments: list[SubsetAssignment] = [
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.TRAINING) for item in items[:30]],
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.VALIDATION) for item in items[30:40]],
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.TESTING) for item in items[40:]],
        ]

        fxt_subset_service.update_subset_assignments(fxt_project_id, assignments)

        # Verify that there are no unassigned items left
        assert (
            db_session.scalar(
                select(func.count())
                .select_from(DatasetItemDB)
                .where(DatasetItemDB.subset == DatasetItemSubset.UNASSIGNED)
            )
            == 0
        )

        # Verify that items have been assigned to corresponding subset using the current distribution values and
        # the new assignments
        distribute_counts = {
            DatasetItemSubset.TRAINING: 30,
            DatasetItemSubset.VALIDATION: 10,
            DatasetItemSubset.TESTING: 10,
        }
        for subset, count in distribute_counts.items():
            assert (
                db_session.scalar(select(func.count()).select_from(DatasetItemDB).where(DatasetItemDB.subset == subset))
                == fxt_default_distribution.get(subset, 0) + count
            )
