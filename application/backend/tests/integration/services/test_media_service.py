# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os.path
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, MediaFormat, Pipeline, Project
from app.services import LabelService, PipelineService, ProjectService, SystemService
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventBus
from app.services.media_service import InvalidImageError, MediaFilters, MediaService


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
def fxt_media_service(
    fxt_projects_dir: Path,
    db_session: Session,
) -> MediaService:
    """Fixture to create a MediaService instance."""
    return MediaService(data_dir=fxt_projects_dir.parent, db_session=db_session)


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
def fxt_project_with_media(fxt_project_with_pipeline, db_session) -> tuple[Project, list[MediaDB]]:
    project, _ = fxt_project_with_pipeline

    configs = [
        {"type": "image", "name": "test1", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
        {"type": "image", "name": "test2", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
        {"type": "image", "name": "test3", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
    ]

    db_media_list = []
    for config in configs:
        db_media = MediaDB(**config)
        db_media.project_id = str(project.id)
        db_media.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_media_list.append(db_media)
    db_session.add_all(db_media_list)
    db_session.flush()
    return project, db_media_list


@pytest.fixture
def fxt_project_with_annotation_status_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all annotation statuses."""
    project, _ = fxt_project_with_pipeline

    # Unannotated items (annotation_data is null) - don't set annotation_data at all
    unannotated_items = [
        DatasetItemDB(
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    # Reviewed items (annotation_data is not null and user_reviewed is True)
    reviewed_items = [
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    # To review items (annotation_data is not null and user_reviewed is False)
    to_review_items = [
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    db_dataset_items = []
    for list, name in zip(
        [unannotated_items, reviewed_items, to_review_items], ["unannotated", "reviewed", "to_review"]
    ):
        for idx, dataset_item in enumerate(list):
            db_media = MediaDB(
                type="image",
                name=f"{name}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()

            dataset_item.id = db_media.id
            db_dataset_items.append(dataset_item)

    db_session.add_all(db_dataset_items)
    db_session.flush()

    # Link labels to annotated dataset items
    for item in [*reviewed_items, *to_review_items]:
        db_session.add(DatasetItemLabelDB(dataset_item_id=item.id, label_id=str(project.task.labels[0].id)))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_labeled_dataset_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture to create a project with multiple labeled dataset items for testing label filtering."""
    project, _ = fxt_project_with_pipeline

    # Ensure we have at least 2 labels
    assert len(project.task.labels) >= 2, "Project must have at least 2 labels for this fixture"

    label_0_id = str(project.task.labels[0].id)
    label_1_id = str(project.task.labels[1].id)

    configs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        # Item 0: No annotations
        (
            {"type": "image", "name": "item_no_labels", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
            {"subset": "unassigned"},
        ),
        # Item 1: Has label_0
        (
            {
                "type": "image",
                "name": "item_label_0",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [{"labels": [{"id": label_0_id}], "shape": {"type": "full_image"}}],
            },
        ),
        # Item 2: Has label_1
        (
            {
                "type": "image",
                "name": "item_label_1",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [{"labels": [{"id": label_1_id}], "shape": {"type": "full_image"}}],
            },
        ),
        # Item 3: Has both label_0 and label_1
        (
            {
                "type": "image",
                "name": "item_both_labels",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [
                    {
                        "labels": [{"id": label_0_id}],
                        "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    },
                    {
                        "labels": [{"id": label_1_id}],
                        "shape": {"type": "rectangle", "x": 20, "y": 20, "width": 10, "height": 10},
                    },
                ],
            },
        ),
    ]

    db_dataset_items = []
    for index, (media_config, dataset_item_config) in enumerate(configs):
        media = MediaDB(**media_config)
        media.project_id = str(project.id)
        db_session.add(media)
        db_session.flush()

        dataset_item = DatasetItemDB(**dataset_item_config)
        dataset_item.id = str(media.id)
        dataset_item.project_id = str(project.id)
        dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_dataset_items.append(dataset_item)
    db_session.add_all(db_dataset_items)
    db_session.flush()

    # Link labels to dataset items
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[1].id, label_id=label_0_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[2].id, label_id=label_1_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[3].id, label_id=label_0_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[3].id, label_id=label_1_id))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_subset_items(fxt_project_with_pipeline, db_session) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all subset types."""
    project, _ = fxt_project_with_pipeline

    # Unassigned items
    unassigned_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-02T00:00:00Z"),
        ),
    ]

    # Training items
    training_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-03T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-04T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-05T00:00:00Z"),
        ),
    ]

    # Validation items
    validation_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-06T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-07T00:00:00Z"),
        ),
    ]

    # Testing items
    testing_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.TESTING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-08T00:00:00Z"),
        ),
    ]
    db_dataset_items = []
    for list in [unassigned_items, training_items, validation_items, testing_items]:
        for idx, dataset_item in enumerate(list):
            db_media = MediaDB(
                type="image",
                name=f"{dataset_item.subset}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-08T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()
            dataset_item.id = db_media.id
            db_dataset_items.append(dataset_item)

    db_session.add_all(db_dataset_items)
    db_session.flush()

    return project, db_dataset_items


class TestMediaServiceIntegration:
    """Integration tests for MediaService."""

    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("format", ["jpg", "png"])
    def test_create_media(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
        format: MediaFormat,
        use_pipeline_source: bool,
    ) -> None:
        """Test creating a media."""
        image = Image.new("RGB", (1024, 768))

        project, pipeline = fxt_project_with_pipeline

        created_media = fxt_media_service.create_image(
            project=project,
            name="test",
            format=format,
            data=image,
            source_id=pipeline.source_id if use_pipeline_source else None,
        )

        media = db_session.get(MediaDB, str(created_media.id))
        assert media is not None
        assert (
            media.id == str(created_media.id)
            and media.project_id == str(project.id)
            and media.type == "image"
            and media.name == "test"
            and media.format == format
            and media.width == 1024
            and media.height == 768
        )
        if use_pipeline_source:
            assert media.source_id == str(pipeline.source_id)
        else:
            assert media.source_id is None

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.{format}"
        assert os.path.exists(binary_file_path)
        assert created_media.size == os.path.getsize(binary_file_path)

        thumbnail_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}-thumb.jpg"
        assert os.path.exists(thumbnail_file_path)

    def test_create_media_invalid_image(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ) -> None:
        """Test creating a media with invalid image."""
        project, _ = fxt_project_with_pipeline

        with pytest.raises(InvalidImageError):
            fxt_media_service.create_image(
                project=project,
                name="test",
                format="jpg",
                data=BytesIO(b"123"),
            )

    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_count_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test counting media."""
        project, db_media_list = fxt_project_with_media

        count = fxt_media_service.count_media(project=project, start_date=start_date, end_date=end_date)

        assert count == 0 if start_date_out_of_range or end_date_out_of_range else len(db_media_list)

    @pytest.mark.parametrize("limit, limit_out_of_range", [(10, False), (0, True)])
    @pytest.mark.parametrize("offset, offset_out_of_range", [(0, False), (10, True)])
    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_list_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        limit,
        limit_out_of_range,
        offset,
        offset_out_of_range,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test listing media."""
        project, db_media_list = fxt_project_with_media

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        assert (
            len(media_list) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(db_media_list)
        )

    def test_get_media_by_id(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media by ID."""
        project, db_media_list = fxt_project_with_media

        fetched_media = fxt_media_service.get_media_by_id(project_id=project.id, media_id=UUID(db_media_list[0].id))

        assert str(fetched_media.id) == db_media_list[0].id and fetched_media.name == db_media_list[0].name

    def test_get_media_by_id_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a non-existent media raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_by_id(project_id=project.id, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_media_binary_path_by_id(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media binary path by ID."""
        project, db_media_list = fxt_project_with_media

        media_binary_path = fxt_media_service.get_media_binary_path_by_id(
            project_id=project.id, media_id=UUID(db_media_list[0].id)
        )

        assert (
            media_binary_path
            == tmp_path / f"projects/{str(project.id)}/dataset/{db_media_list[0].id}.{db_media_list[0].format}"
        )

    def test_get_media_binary_path_by_id_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a non-existent media binary path raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_binary_path_by_id(project_id=project.id, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_media_thumbnail_path_by_id(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media thumbnail path by ID."""
        project, db_media_list = fxt_project_with_media

        media_binary_path = fxt_media_service.get_media_thumbnail_path_by_id(
            project=project, media_id=UUID(db_media_list[0].id)
        )

        assert media_binary_path == tmp_path / f"projects/{str(project.id)}/dataset/{db_media_list[0].id}-thumb.jpg"

    def test_get_media_thumbnail_path_by_id_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a non-existent media thumbnail path raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_thumbnail_path_by_id(project=project, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_generate_media_thumbnail(
        self,
        tmp_path: Path,
        fxt_projects_dir: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test generating a dataset item thumbnail returns a PIL Image."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[0]

        # Create the dataset directory and a test image file
        dataset_dir = tmp_path / fxt_projects_dir / str(project.id) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        image_path = dataset_dir / f"{media.id}.{media.format}"
        test_image = Image.new("RGB", (1024, 768), color="red")
        test_image.save(image_path)

        thumbnail = fxt_media_service.generate_media_thumbnail(project=project, media_id=UUID(media.id))

        assert isinstance(thumbnail, Image.Image)
        assert thumbnail.width == 64
        assert thumbnail.height == 64

    def test_generate_media_thumbnail_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test generating a thumbnail for a non-existent dataset item raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.generate_media_thumbnail(project=project, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        fxt_projects_dir: Path,
        db_session: Session,
    ):
        """Test deleting a media."""
        project, db_media_list = fxt_project_with_media

        dataset_dir = fxt_projects_dir / str(project.id) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        binary_path = dataset_dir / f"{db_media_list[0].id}.{db_media_list[0].format}"
        binary_path.touch()
        assert os.path.exists(binary_path)

        thumbnail_path = dataset_dir / f"{db_media_list[0].id}-thumb.jpg"
        thumbnail_path.touch()
        assert os.path.exists(thumbnail_path)

        """Test deleting a media."""
        fxt_media_service.delete_media(project=project, media_id=UUID(db_media_list[0].id))

        assert db_session.get(MediaDB, db_media_list[0].id) is None
        assert not os.path.exists(binary_path)
        assert not os.path.exists(thumbnail_path)

    def test_delete_media_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test deleting a non-existent media raises error."""
        project, db_media_list = fxt_project_with_media

        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.delete_media(project=project, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    @pytest.mark.parametrize(
        "annotation_status, expected_count",
        [
            (None, 7),  # All items
            ("unannotated", 2),  # 2 unannotated items
            ("reviewed", 3),  # 3 items with user_reviewed=True
            ("to_review", 4),  # 2 unannotated items + 2 items with user_reviewed=False
        ],
    )
    def test_count_media_with_annotation_status(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: str | None,
        expected_count: int,
    ) -> None:
        """Test counting media with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        count = fxt_media_service.count_media(project=project, annotation_status=annotation_status)

        assert count == expected_count

    @pytest.mark.parametrize(
        "annotation_status, expected_names",
        [
            (None, ["unannotated1", "unannotated2", "reviewed1", "reviewed2", "reviewed3", "to_review1", "to_review2"]),
            (DatasetItemAnnotationStatus.UNANNOTATED, ["unannotated1", "unannotated2"]),
            (DatasetItemAnnotationStatus.REVIEWED, ["reviewed1", "reviewed2", "reviewed3"]),
            (DatasetItemAnnotationStatus.TO_REVIEW, ["unannotated1", "unannotated2", "to_review1", "to_review2"]),
        ],
    )
    def test_list_media_with_annotation_status(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_names: list[str],
    ) -> None:
        """Test listing media with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=annotation_status,
            ),
        )

        assert len(media_list) == len(expected_names)
        actual_names = sorted([media.name for media in media_list])
        assert actual_names == sorted(expected_names)

    @pytest.mark.parametrize(
        "annotation_status, limit, offset, expected_count",
        [
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 0, 1),  # First page of unannotated
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 1, 1),  # Second page of unannotated
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 2, 0),  # Beyond available unannotated items
            (DatasetItemAnnotationStatus.REVIEWED, 2, 0, 2),  # First page of reviewed
            (DatasetItemAnnotationStatus.REVIEWED, 2, 2, 1),  # Second page of reviewed (only 1 left)
            (DatasetItemAnnotationStatus.TO_REVIEW, 10, 0, 4),  # All items with user_reviewed=False
        ],
    )
    def test_list_media_with_annotation_status_pagination(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing media with annotation_status filter and pagination."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                annotation_status=annotation_status,
            ),
        )

        assert len(media_list) == expected_count

    def test_list_media_annotation_status_combined_with_dates(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test annotation_status filter combined with date filters."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        # All reviewed items within date range
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                start_date=datetime.fromisoformat("2025-01-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-02-02T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.REVIEWED,
            ),
        )
        assert len(media_list) == 3

        # No items outside date range
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                start_date=datetime.fromisoformat("2025-03-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-03-31T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.UNANNOTATED,
            ),
        )
        assert len(media_list) == 0

    def test_annotation_status_filter_verifies_data_correctness(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that annotation_status filter returns items with correct properties."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        unannotated_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=DatasetItemAnnotationStatus.UNANNOTATED,
            ),
        )
        assert len(unannotated_items) == 2

        reviewed_media = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=DatasetItemAnnotationStatus.REVIEWED,
            ),
        )
        assert len(reviewed_media) == 3

        to_review_media = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=DatasetItemAnnotationStatus.TO_REVIEW,
            ),
        )
        assert len(to_review_media) == 4

    def test_list_media_filter_by_single_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Filter by label_0 - should return media 1 and 3 (item_label_0 and item_both_labels)
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[label_0_id],
            ),
        )

        assert len(media_list) == 2
        item_names = {item.name for item in media_list}
        assert item_names == {"item_label_0", "item_both_labels"}

    def test_list_media_filter_by_multiple_labels(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Filter by label_0 OR label_1 - should return items 1, 2, and 3
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[label_0_id, label_1_id],
            ),
        )

        assert len(media_list) == 3
        item_names = {item.name for item in media_list}
        assert item_names == {"item_label_0", "item_label_1", "item_both_labels"}

    def test_list_media_filter_by_nonexistent_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by a nonexistent label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        nonexistent_label_id = uuid4()

        # Filter by nonexistent label - should return empty list
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[nonexistent_label_id],
            ),
        )

        assert len(media_list) == 0

    def test_count_media_filter_by_single_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting media filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Count media with label_0 - should return 2
        count = fxt_media_service.count_media(
            project=project,
            label_ids=[label_0_id],
        )

        assert count == 2

    def test_count_media_filter_by_multiple_labels(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting media filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Count media with label_0 OR label_1 - should return 3
        count = fxt_media_service.count_media(
            project=project,
            label_ids=[label_0_id, label_1_id],
        )

        assert count == 3

    def test_list_media_no_label_filter(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media without label filter returns all items."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items

        # No filter - should return all 4 items
        media_list = fxt_media_service.list_media(project_id=project.id)

        assert len(media_list) == 4
        item_names = {item.name for item in media_list}
        assert item_names == {"item_no_labels", "item_label_0", "item_label_1", "item_both_labels"}

    @pytest.mark.parametrize(
        "subset, expected_count",
        [
            (None, 8),  # All items
            ("unassigned", 2),  # 2 unassigned items
            ("training", 3),  # 3 training items
            ("validation", 2),  # 2 validation items
            ("testing", 1),  # 1 testing item
        ],
    )
    def test_count_media_with_subset(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        expected_count: int,
    ) -> None:
        """Test counting media with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        count = fxt_media_service.count_media(project=project, subset=subset)

        assert count == expected_count

    @pytest.mark.parametrize(
        "subset, expected_names",
        [
            (
                None,
                [
                    "unassigned1",
                    "unassigned2",
                    "training1",
                    "training2",
                    "training3",
                    "validation1",
                    "validation2",
                    "testing1",
                ],
            ),
            ("unassigned", ["unassigned1", "unassigned2"]),
            ("training", ["training1", "training2", "training3"]),
            ("validation", ["validation1", "validation2"]),
            ("testing", ["testing1"]),
        ],
    )
    def test_list_media_with_subset(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        expected_names: list[str],
    ) -> None:
        """Test listing media with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subset=subset,
            ),
        )

        assert len(media_list) == len(expected_names)
        actual_names = sorted([item.name for item in media_list])
        assert actual_names == sorted(expected_names)

    @pytest.mark.parametrize(
        "subset, limit, offset, expected_count",
        [
            ("unassigned", 1, 0, 1),  # First page of unassigned
            ("unassigned", 1, 1, 1),  # Second page of unassigned
            ("unassigned", 1, 2, 0),  # Beyond available unassigned items
            ("training", 2, 0, 2),  # First page of training
            ("training", 2, 2, 1),  # Second page of training (only 1 left)
            ("validation", 10, 0, 2),  # All validation items
            ("testing", 10, 0, 1),  # All testing items
        ],
    )
    def test_list_media_with_subset_pagination(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing media with subset filter and pagination."""
        project, db_dataset_items = fxt_project_with_subset_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                subset=subset,
            ),
        )

        assert len(media_list) == expected_count

    def test_subset_filter_verifies_data_correctness(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that subset filter returns media with correct subset values."""
        project, db_dataset_items = fxt_project_with_subset_items

        # Unassigned items should have subset=unassigned
        unassigned_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subset="unassigned",
            ),
        )
        assert len(unassigned_items) == 2

        # Training items should have subset=training
        training_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subset="training",
            ),
        )
        assert len(training_items) == 3

        # Validation items should have subset=validation
        validation_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subset="validation",
            ),
        )
        assert len(validation_items) == 2

        # Testing items should have subset=testing
        testing_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subset="testing",
            ),
        )
        assert len(testing_items) == 1
