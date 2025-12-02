# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, LabelDB, ModelRevisionDB, PipelineDB, ProjectDB
from app.models import DatasetItemSubset


class ProjectTestDataFactory:
    """
    Factory for creating test projects with related entities in the database.

    This factory provides an interface for building complex test data scenarios involving projects and their related
    entities (pipelines, models, labels, datasets). It handles proper relationship setup and database persistence.

    The factory uses a builder pattern where you chain method calls to configure the project structure,
    then call `build()` to persist everything to the database.

    Usage:
        Basic project:
        >>> factory = ProjectTestDataFactory(db_session)
        >>> project = factory.with_project(ProjectDB(name="Test")).build()

        Project with pipeline and models:
        >>> project = (
        ...     ProjectTestDataFactory(db_session)
        ...     .with_project(ProjectDB(name="Test"))
        ...     .with_pipeline(is_running=True)
        ...     .with_models([ModelRevisionDB(...)])
        ...     .build()
        ... )

        Project with labeled dataset:
        >>> label = LabelDB(name="cat")
        >>> project = (
        ...     ProjectTestDataFactory(db_session)
        ...     .with_project(ProjectDB(name="Test"))
        ...     .with_label(label)
        ...     .with_dataset_items({DatasetItemSubset.TRAINING: 10})
        ...     .with_item_labels(label)
        ...     .build()
        ... )

    Args:
        db_session: SQLAlchemy session for database operations.

    Raises:
        ValueError: If methods are called in invalid order (e.g., adding entities
                   before setting a project, or linking labels before adding items).
    """

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self._project: ProjectDB | None = None
        self._pipeline: PipelineDB | None = None
        self._model_revisions: list[ModelRevisionDB] = []
        self._labels: list[LabelDB] = []
        self._dataset_items: list[DatasetItemDB] = []
        self._item_labels: list[DatasetItemLabelDB] = []

    def with_project(self, project: ProjectDB) -> "ProjectTestDataFactory":
        """Add a project to the builder."""
        self._project = project
        return self

    def with_pipeline(
        self,
        is_running: bool = False,
        model_id: str | None = None,
        source_id: str | None = None,
        sink_id: str | None = None,
    ) -> "ProjectTestDataFactory":
        """Add a pipeline to the project."""
        if not self._project:
            raise ValueError("Project must be set before adding a pipeline")

        self._pipeline = PipelineDB(
            project_id=self._project.id,
            is_running=is_running,
            model_revision_id=model_id,
            source_id=source_id,
            sink_id=sink_id,
        )
        return self

    def with_models(self, model_revisions: list[ModelRevisionDB]) -> "ProjectTestDataFactory":
        """Add models to the project."""
        if not self._project:
            raise ValueError("Project must be set before adding models")

        for model in model_revisions:
            model.project_id = self._project.id
        self._model_revisions.extend(model_revisions)
        return self

    def with_label(self, label: LabelDB) -> "ProjectTestDataFactory":
        """Add a label to the project."""
        if not self._project:
            raise ValueError("Project must be set before adding labels")
        label.project_id = self._project.id
        self._labels.append(label)
        return self

    def with_dataset_items(self, subset_distribution: dict[DatasetItemSubset, int]) -> "ProjectTestDataFactory":
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

    def with_item_labels(self, label: LabelDB) -> "ProjectTestDataFactory":
        """Link all dataset items to a label."""
        if not self._dataset_items:
            raise ValueError("Dataset items must be added before linking labels")

        item_labels = [DatasetItemLabelDB(dataset_item_id=item.id, label_id=label.id) for item in self._dataset_items]
        self._item_labels.extend(item_labels)
        return self

    def with_data_policies(self, data_policies: list[dict]) -> "ProjectTestDataFactory":
        """Set data collection policy for the project."""
        if not self._pipeline:
            raise ValueError("Pipeline must be set before adding data policies")
        self._pipeline.data_collection_policies = data_policies
        return self

    def build(self) -> ProjectDB:
        """Build and persist the project with all entities."""
        if not self._project:
            raise ValueError("Project must be set before building")

        self.db_session.add(self._project)
        self.db_session.flush()

        if self._model_revisions:
            self.db_session.add_all(self._model_revisions)
            self.db_session.flush()

        if self._pipeline:
            self.db_session.add(self._pipeline)
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
