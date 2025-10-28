# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.models import DatasetItemSubset
from app.db.schema import DatasetItemDB, DatasetItemLabelDB, LabelDB, ProjectDB


class ProjectBuilder:
    """Builder for creating test projects with related entities."""

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self._project: ProjectDB | None = None
        self._labels: list[LabelDB] = []
        self._dataset_items: list[DatasetItemDB] = []
        self._item_labels: list[DatasetItemLabelDB] = []

    def with_project(self, project: ProjectDB) -> "ProjectBuilder":
        """Add a project to the builder."""
        self._project = project
        return self

    def with_label(self, label: LabelDB) -> "ProjectBuilder":
        """Add a label to the project."""
        if not self._project:
            raise ValueError("Project must be set before adding labels")
        label.project_id = self._project.id
        self._labels.append(label)
        return self

    def with_dataset_items(self, subset_distribution: dict[DatasetItemSubset, int]) -> "ProjectBuilder":
        """Add dataset items with specified subset distribution."""
        if not self._project:
            raise ValueError("Project must be set before adding dataset items")

        items = [
            DatasetItemDB(
                id=str(uuid4()),
                name=f"test_item_{idx}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                subset=str(subset),
                project_id=self._project.id,
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            for subset, count in subset_distribution.items()
            for idx in range(count)
        ]
        self._dataset_items.extend(items)
        return self

    def with_item_labels(self, label: LabelDB) -> "ProjectBuilder":
        """Link all dataset items to a label."""
        if not self._dataset_items:
            raise ValueError("Dataset items must be added before linking labels")

        item_labels = [DatasetItemLabelDB(dataset_item_id=item.id, label_id=label.id) for item in self._dataset_items]
        self._item_labels.extend(item_labels)
        return self

    def build(self) -> ProjectDB:
        """Build and persist the project with all entities."""
        if not self._project:
            raise ValueError("Project must be set before building")

        self.db_session.add(self._project)
        self.db_session.flush()

        if self._labels:
            self.db_session.add_all(self._labels)
            self.db_session.flush()

        if self._dataset_items:
            self.db_session.add_all(self._dataset_items)

        if self._item_labels:
            self.db_session.add_all(self._item_labels)

        self.db_session.flush()
        return self._project
