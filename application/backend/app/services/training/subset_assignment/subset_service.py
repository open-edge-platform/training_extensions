# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.models import DatasetItemSubset
from app.repositories import DatasetItemRepository

from .distribution import SubsetDistribution
from .models import DatasetItemWithLabels, SubsetAssignment


class SubsetService:
    def get_unassigned_items_with_labels(self, project_id: UUID, db_session: Session) -> list[DatasetItemWithLabels]:
        """Retrieve all unassigned dataset items for a given project."""
        repo = DatasetItemRepository(project_id=str(project_id), db=db_session)
        unassigned_items_db = repo.list_unassigned_items()

        items_dict = defaultdict(set)
        for label in unassigned_items_db:
            items_dict[label.dataset_item_id].add(UUID(label.label_id))

        return [DatasetItemWithLabels(item_id=UUID(item_id), labels=labels) for item_id, labels in items_dict.items()]

    def get_subset_distribution(self, project_id: UUID, db_session: Session) -> SubsetDistribution:
        """Get distribution of dataset items across subsets."""
        repo = DatasetItemRepository(project_id=str(project_id), db=db_session)
        results = repo.get_subset_distribution()

        return SubsetDistribution(counts={DatasetItemSubset(subset): count for subset, count in results.items()})

    def update_subset_assignments(
        self, project_id: UUID, assignments: list[SubsetAssignment], db_session: Session
    ) -> None:
        """Update subset assignments for dataset items."""
        repo = DatasetItemRepository(project_id=str(project_id), db=db_session)

        assignments_by_subset = defaultdict(set)
        for assignment in assignments:
            assignments_by_subset[assignment.subset].add(str(assignment.item_id))

        for subset, item_ids in assignments_by_subset.items():
            logger.info("Updating subset assignments for {}: {} items", subset, len(item_ids))
            repo.set_subset(obj_ids=item_ids, subset=subset)
