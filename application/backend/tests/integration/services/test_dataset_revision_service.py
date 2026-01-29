# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import datumaro.experimental as dm
import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetRevisionDB, MediaDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Pipeline, Project
from app.services import (
    DatasetRevisionService,
    DatasetService,
    LabelService,
    MediaService,
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
def fxt_media_service(fxt_projects_dir: Path, db_session: Session) -> MediaService:
    """Fixture to create a MediaService instance."""
    return MediaService(data_dir=fxt_projects_dir.parent, db_session=db_session)


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
    fxt_label_service: LabelService,
    fxt_media_service: MediaService,
    db_session: Session,
) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(label_service=fxt_label_service, media_service=fxt_media_service, db_session=db_session)


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
def fxt_project_with_subset_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[tuple[MediaDB, DatasetItemDB]]]:
    """Fixture with dataset items covering all subset types."""
    project, _ = fxt_project_with_pipeline

    distribution = [
        (DatasetItemSubset.UNASSIGNED, 2),
        (DatasetItemSubset.TRAINING, 3),
        (DatasetItemSubset.VALIDATION, 2),
        (DatasetItemSubset.TESTING, 1),
    ]

    db_media_and_dataset_items = []
    for subset, item_count in distribution:
        for idx in range(item_count):
            db_media = MediaDB(
                type="image",
                name=f"{subset.value}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()

            dataset_item = DatasetItemDB(
                subset=subset,
                user_reviewed=False,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            dataset_item.id = db_media.id

            db_session.add(dataset_item)
            db_session.flush()

            db_media_and_dataset_items.append((db_media, dataset_item))

    return project, db_media_and_dataset_items


@pytest.fixture
def fxt_project_with_subset_items_on_disk(
    fxt_projects_dir, fxt_db_labels, fxt_project_with_pipeline, db_session, fxt_label_service
) -> tuple[Project, list[tuple[MediaDB, DatasetItemDB]]]:
    """Fixture with dataset items covering all subset types and annotation data."""
    project, _ = fxt_project_with_pipeline

    # Get the first label for annotation
    label = fxt_db_labels[0]
    label_id = str(label.id)

    def annotation():
        # Rectangle annotation with one label
        return [
            {
                "shape": {
                    "type": "rectangle",
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                },
                "labels": [{"id": label_id, "name": label.name, "color": label.color}],
            }
        ]

    def make_item(name: str, subset: DatasetItemSubset, created_at: datetime):
        db_media = MediaDB(
            type="image",
            name=name,
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
            created_at=created_at,
        )
        db_session.add(db_media)
        db_session.flush()

        dataset_item = DatasetItemDB(
            subset=subset.name.lower(),
            user_reviewed=True,
            project_id=str(project.id),
            created_at=created_at,
            annotation_data=annotation(),
        )
        dataset_item.id = db_media.id

        db_session.add(dataset_item)
        db_session.flush()

        return db_media, dataset_item

    media_and_dataset_items = [
        make_item("unassigned1", DatasetItemSubset.UNASSIGNED, datetime.fromisoformat("2025-02-01T00:00:00Z")),
        make_item("unassigned2", DatasetItemSubset.UNASSIGNED, datetime.fromisoformat("2025-02-02T00:00:00Z")),
        make_item("training1", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-03T00:00:00Z")),
        make_item("training2", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-04T00:00:00Z")),
        make_item("training3", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-05T00:00:00Z")),
        make_item("validation1", DatasetItemSubset.VALIDATION, datetime.fromisoformat("2025-02-06T00:00:00Z")),
        make_item("validation2", DatasetItemSubset.VALIDATION, datetime.fromisoformat("2025-02-07T00:00:00Z")),
        make_item("testing1", DatasetItemSubset.TESTING, datetime.fromisoformat("2025-02-08T00:00:00Z")),
    ]

    # Create images directory
    images_dir = fxt_projects_dir / str(project.id) / "dataset"
    images_dir.mkdir(parents=True, exist_ok=True)

    def create_item(item: MediaDB) -> None:
        # Create dummy image file
        image_path = images_dir / f"{item.id}.{item.format}"
        image_path.write_bytes(b"\x00")  # 1-byte dummy file

    for media_db, _ in media_and_dataset_items:
        create_item(media_db)

    return project, media_and_dataset_items


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
        project, _ = fxt_project_with_subset_items
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
        project, _ = fxt_project_with_subset_items
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
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test getting a dataset revision with wrong project ID raises error."""
        project, _ = fxt_project_with_subset_items
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

    def test_count_items_by_subset(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test counting dataset items by subset."""
        # Create non-empty dataset in memory
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, annotation_status=DatasetItemAnnotationStatus.REVIEWED
        )
        assert len(dataset) > 0

        # Create a non-empty parquet file on disk
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )
        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert revision_path.exists()
        assert (revision_path / "data.parquet").exists()

        # Count items in each subset
        counts = fxt_dataset_revision_service.count_items_by_subset(project.id, revision_id)

        # Calculate expected counts from fixture data
        expected_counts: dict[str, int] = {}
        for _, dataset_item in media_and_dataset_items:
            subset_name = dataset_item.subset if dataset_item.subset is not None else DatasetItemSubset.UNASSIGNED.name
            expected_counts[subset_name] = expected_counts.get(subset_name, 0) + 1
        expected_total = sum(expected_counts.values())

        # Verify counts match expected values from fixture
        for subset_name, expected_count in expected_counts.items():
            assert counts[subset_name] == expected_count
        assert counts["total"] == expected_total

    def test_delete_dataset_revision_files(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files."""
        project, _ = fxt_project_with_subset_items
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
        project, _ = fxt_project_with_subset_items
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

    def test_load_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test loading a dataset revision as a Datumaro dataset."""
        project, _ = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Load the revision
        loaded_dataset = fxt_dataset_revision_service.load_revision(
            project_id=project.id, dataset_revision_id=revision_id
        )

        # Verify it returns a Datumaro dataset
        assert isinstance(loaded_dataset, dm.Dataset)

    def test_load_revision_files_deleted(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test loading a revision with deleted files raises error."""
        project, _ = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Delete the revision files
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Try to load the revision - should raise error
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.load_revision(project_id=project.id, dataset_revision_id=revision_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(revision_id)
