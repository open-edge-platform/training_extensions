# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetRevisionDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Pipeline, Project
from app.services import (
    DatasetRevisionService,
    DatasetService,
    LabelService,
    PipelineService,
    ProjectService,
    SystemService,
)
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus


@pytest.fixture
def fxt_event_bus() -> EventBus:
    """Fixture to create a EventBus instance."""
    return EventBus()


@pytest.fixture
def fxt_system_service() -> SystemService:
    """Fixture to create a SystemService instance."""
    return SystemService()


@pytest.fixture
def fxt_pipeline_service(
    fxt_event_bus: EventBus, db_session: Session, fxt_system_service: SystemService
) -> PipelineService:
    """Fixture to create a PipelineService instance."""
    return PipelineService(event_bus=fxt_event_bus, db_session=db_session, system_service=fxt_system_service)


@pytest.fixture
def fxt_label_service(db_session: Session) -> LabelService:
    """Fixture to create a LabelService instance."""
    return LabelService(db_session=db_session)


@pytest.fixture
def fxt_project_service(
    fxt_projects_dir: Path, db_session: Session, fxt_pipeline_service: PipelineService, fxt_label_service: LabelService
) -> ProjectService:
    """Fixture to create a ProjectService instance."""
    return ProjectService(
        fxt_projects_dir.parent,
        db_session=db_session,
        pipeline_service=fxt_pipeline_service,
        label_service=fxt_label_service,
    )


@pytest.fixture
def fxt_dataset_revision_service(
    fxt_projects_dir: Path,
    db_session: Session,
) -> DatasetRevisionService:
    """Fixture to create a DatasetRevisionService instance."""
    return DatasetRevisionService(data_dir=fxt_projects_dir.parent, db_session=db_session)


@pytest.fixture
def fxt_dataset_service(
    fxt_projects_dir: Path,
    fxt_label_service: LabelService,
    db_session: Session,
) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(fxt_projects_dir.parent, fxt_label_service, db_session=db_session)


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects,
    fxt_db_labels,
    fxt_project_service,
    fxt_pipeline_service,
    fxt_db_sources,
    fxt_db_sinks,
    fxt_db_models,
    db_session,
) -> tuple[Project, Pipeline]:
    """Fixture to create a Project."""

    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    db_model = fxt_db_models[0]
    db_model.project_id = db_project.id
    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all([db_model, *fxt_db_labels])
    db_session.flush()

    db_pipeline = PipelineDB(project_id=db_project.id)
    db_pipeline.source = fxt_db_sources[0]
    db_pipeline.sink = fxt_db_sinks[0]
    db_pipeline.model_revision = db_model
    db_session.add(db_pipeline)
    db_session.flush()

    return fxt_project_service.get_project_by_id(UUID(db_project.id)), fxt_pipeline_service.get_pipeline_by_id(
        UUID(db_project.id)
    )


@pytest.fixture
def fxt_project_with_subset_items(fxt_project_with_pipeline, db_session) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all subset types."""
    project, _ = fxt_project_with_pipeline

    # Unassigned items
    unassigned_items = [
        DatasetItemDB(
            name="unassigned1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            name="unassigned2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-02T00:00:00Z"),
        ),
    ]

    # Training items
    training_items = [
        DatasetItemDB(
            name="training1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-03T00:00:00Z"),
        ),
        DatasetItemDB(
            name="training2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-04T00:00:00Z"),
        ),
        DatasetItemDB(
            name="training3",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-05T00:00:00Z"),
        ),
    ]

    # Validation items
    validation_items = [
        DatasetItemDB(
            name="validation1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-06T00:00:00Z"),
        ),
        DatasetItemDB(
            name="validation2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-07T00:00:00Z"),
        ),
    ]

    # Testing items
    testing_items = [
        DatasetItemDB(
            name="testing1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset=DatasetItemSubset.TESTING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-08T00:00:00Z"),
        ),
    ]

    db_dataset_items = [*unassigned_items, *training_items, *validation_items, *testing_items]
    db_session.add_all(db_dataset_items)
    db_session.flush()

    return project, db_dataset_items


class TestDatasetRevisionServiceIntegration:
    """Integration tests for DatasetRevisionService."""

    def test_save_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test saving a dataset revision."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Verify that a revision entry was created
        assert db_session.get(DatasetRevisionDB, str(revision_id)) is not None
        assert (fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id) / "data.parquet").exists()

    def test_get_dataset_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test getting a dataset revision."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Now get the revision
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        assert revision is not None
        assert revision.id == revision_id
        assert revision.project_id == project.id
        assert revision.name == f"Dataset ({str(revision.id).split('-')[0]})"
        assert revision.files_deleted is False

    def test_get_dataset_revision_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test getting a non-existent dataset revision raises error."""
        project, _ = fxt_project_with_subset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_revision_wrong_project(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test getting a dataset revision with wrong project ID raises error."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision for the project
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Try to get the revision with a different project ID
        wrong_project_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision(project_id=wrong_project_id, revision_id=revision_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(revision_id)

    def test_delete_dataset_revision_files(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert revision_path.exists()
        assert (revision_path / "data.parquet").exists()

        # Delete the revision files
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify the files were deleted
        assert not revision_path.exists()

        # Verify the database record is marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True

    def test_delete_dataset_revision_files_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test deleting files for a non-existent dataset revision raises error."""
        project, _ = fxt_project_with_subset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.delete_dataset_revision_files(
                project_id=project.id, revision_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_revision_files_already_deleted(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files that are already deleted is idempotent."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Delete the revision files once
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify files are deleted
        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert not revision_path.exists()

        # Delete again - should not raise an error
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify it's still marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True

    def test_delete_dataset_revision_files_no_directory(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files when directory doesn't exist."""
        project, _ = fxt_project_with_subset_items

        # Create a revision record in the database without creating files
        revision_id = uuid4()
        db_revision = DatasetRevisionDB(
            id=str(revision_id),
            project_id=str(project.id),
            name=f"Dataset ({str(revision_id).split('-')[0]})",
            files_deleted=False,
        )
        db_session.add(db_revision)
        db_session.flush()

        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert not revision_path.exists()

        # Delete the revision files - should not raise an error even though directory doesn't exist
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify it's marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True
