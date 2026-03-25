# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from uuid import UUID

from loguru import logger

from app.repositories import DatasetItemRepository
from app.services import BaseSessionManagedService

from .models import DatasetItemWithLabels, SubsetAssignment


class SubsetService(BaseSessionManagedService):
    def get_unassigned_items_with_labels(self, project_id: UUID) -> list[DatasetItemWithLabels]:
        """Retrieve all unassigned dataset items for a given project."""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        unassigned_items_db = repo.list_unassigned_items()

        items_dict = defaultdict(set)
        for label in unassigned_items_db:
            items_dict[label.dataset_item_id].add(UUID(label.label_id))

        return [DatasetItemWithLabels(item_id=UUID(item_id), labels=labels) for item_id, labels in items_dict.items()]

    def has_all_subsets_assigned(self, project_id: UUID) -> bool:
        """Return True if there is at least one dataset item for each of TRAINING, VALIDATION, and TESTING subsets."""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        return repo.has_all_subsets_assigned()

    def update_subset_assignments(self, project_id: UUID, assignments: list[SubsetAssignment]) -> None:
        """Update subset assignments for dataset items."""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)

        assignments_by_subset = defaultdict(set)
        for assignment in assignments:
            assignments_by_subset[assignment.subset].add(str(assignment.item_id))

        for subset, item_ids in assignments_by_subset.items():
            logger.info("Updating subset assignments for {}: {} items", subset, len(item_ids))
            repo.set_subset(obj_ids=item_ids, subset=subset)
