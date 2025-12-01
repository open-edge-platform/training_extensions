# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os.path
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, DatasetRevisionDB, PipelineDB
from app.models import (
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    LabelReference,
    Pipeline,
    Project,
    Rectangle,
)
from app.services import LabelService, PipelineService, ProjectService
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_service import (
    DatasetItemFilters,
    DatasetService,
    InvalidImageError,
    SubsetAlreadyAssignedError,
)
from app.services.event.event_bus import EventBus


@pytest.fixture
def fxt_event_bus() -> EventBus:
    """Fixture to create a EventBus instance."""
    return EventBus()


@pytest.fixture
def fxt_pipeline_service(fxt_event_bus: EventBus, db_session: Session) -> PipelineService:
    """Fixture to create a PipelineService instance."""
    return PipelineService(event_bus=fxt_event_bus, db_session=db_session)


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
def fxt_project_with_dataset_items(fxt_project_with_pipeline, db_session) -> tuple[Project, list[DatasetItemDB]]:
    project, _ = fxt_project_with_pipeline

    configs = [
        {"name": "test1", "format": "jpg", "size": 1024, "width": 1024, "height": 768, "subset": "unassigned"},
        {
            "name": "test2",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "subset": "unassigned",
            "annotation_data": [{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
        },
        {"name": "test3", "format": "jpg", "size": 1024, "width": 1024, "height": 768, "subset": "training"},
    ]

    db_dataset_items = []
    for config in configs:
        dataset_item = DatasetItemDB(**config)
        dataset_item.project_id = str(project.id)
        dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_dataset_items.append(dataset_item)
    db_session.add_all(db_dataset_items)
    db_session.flush()
    # Link label to dataset item with annotation
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[1].id, label_id=str(project.task.labels[0].id)))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_annotation_status_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all annotation statuses."""
    project, _ = fxt_project_with_pipeline

    # Unannotated items (annotation_data is null) - don't set annotation_data at all
    unannotated_items = [
        DatasetItemDB(
            name="unannotated1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            name="unannotated2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    # Reviewed items (annotation_data is not null and user_reviewed is True)
    reviewed_items = [
        DatasetItemDB(
            name="reviewed1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            name="reviewed2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            name="reviewed3",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
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
            name="to_review1",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            name="to_review2",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    db_dataset_items = [*unannotated_items, *reviewed_items, *to_review_items]
    db_session.add_all(db_dataset_items)
    db_session.flush()

    # Link labels to annotated dataset items
    for item in [*reviewed_items, *to_review_items]:
        db_session.add(DatasetItemLabelDB(dataset_item_id=item.id, label_id=str(project.task.labels[0].id)))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_annotations() -> Callable[[UUID], list[DatasetItemAnnotation]]:
    def _create_annotations(label_id: UUID) -> list[DatasetItemAnnotation]:
        return [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]

    return _create_annotations


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

    configs = [
        # Item 0: No annotations
        {"name": "item_no_labels", "format": "jpg", "size": 1024, "width": 1024, "height": 768, "subset": "unassigned"},
        # Item 1: Has label_0
        {
            "name": "item_label_0",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "subset": "unassigned",
            "annotation_data": [{"labels": [{"id": label_0_id}], "shape": {"type": "full_image"}}],
        },
        # Item 2: Has label_1
        {
            "name": "item_label_1",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "subset": "unassigned",
            "annotation_data": [{"labels": [{"id": label_1_id}], "shape": {"type": "full_image"}}],
        },
        # Item 3: Has both label_0 and label_1
        {
            "name": "item_both_labels",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
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
    ]

    db_dataset_items = []
    for config in configs:
        dataset_item = DatasetItemDB(**config)
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


class TestDatasetServiceIntegration:
    """Integration tests for DatasetService."""

    @pytest.mark.parametrize("use_pipeline_model", [True, False])
    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("user_reviewed", [True, False])
    @pytest.mark.parametrize("format", ["jpg", "png"])
    def test_create_dataset_item(
        self,
        tmp_path: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
        db_session: Session,
        format: DatasetItemSubset,
        user_reviewed: bool,
        use_pipeline_source: bool,
        use_pipeline_model: bool,
    ) -> None:
        """Test creating a dataset item."""
        image = Image.new("RGB", (1024, 768))

        project, pipeline = fxt_project_with_pipeline
        label_id = project.task.labels[0].id

        created_dataset_item = fxt_dataset_service.create_dataset_item(
            project=project,
            name="test",
            format=format,
            data=image,
            user_reviewed=user_reviewed,
            source_id=pipeline.source_id if use_pipeline_source else None,
            prediction_model_id=pipeline.model_revision_id if use_pipeline_model else None,
            annotations=fxt_annotations(label_id) if not user_reviewed else None,
        )

        dataset_item = db_session.get(DatasetItemDB, str(created_dataset_item.id))
        assert dataset_item is not None
        assert (
            dataset_item.id == str(created_dataset_item.id)
            and dataset_item.project_id == str(project.id)
            and dataset_item.name == "test"
            and dataset_item.format == format
            and dataset_item.width == 1024
            and dataset_item.height == 768
            and dataset_item.user_reviewed == user_reviewed
            and dataset_item.subset == DatasetItemSubset.UNASSIGNED
            and dataset_item.subset_assigned_at is None
        )
        if use_pipeline_source:
            assert dataset_item.source_id == str(pipeline.source_id)
        else:
            assert dataset_item.source_id is None
        if use_pipeline_model:
            assert dataset_item.prediction_model_id == str(pipeline.model_revision_id)
        else:
            assert dataset_item.prediction_model_id is None
        if not user_reviewed:
            assert dataset_item.annotation_data == [
                {
                    "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    "labels": [{"id": str(label_id)}],
                    "confidences": None,
                }
            ]
            assert db_session.get(DatasetItemLabelDB, (str(created_dataset_item.id), str(label_id))) is not None
        else:
            assert dataset_item.annotation_data is None

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_dataset_item.id}.{format}"
        assert os.path.exists(binary_file_path)
        assert created_dataset_item.size == os.path.getsize(binary_file_path)

        thumbnail_file_path = tmp_path / f"projects/{project.id}/dataset/{created_dataset_item.id}-thumb.jpg"
        assert os.path.exists(thumbnail_file_path)

    def test_create_dataset_item_invalid_image(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ) -> None:
        """Test creating a dataset item with invalid image."""
        project, _ = fxt_project_with_pipeline

        with pytest.raises(InvalidImageError):
            fxt_dataset_service.create_dataset_item(
                project=project,
                name="test",
                format="jpg",
                data=BytesIO(b"123"),
                user_reviewed=True,
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
    def test_count_dataset_items(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test counting dataset items."""
        project, db_dataset_items = fxt_project_with_dataset_items

        count = fxt_dataset_service.count_dataset_items(project=project, start_date=start_date, end_date=end_date)

        assert count == 0 if start_date_out_of_range or end_date_out_of_range else len(db_dataset_items)

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
    def test_list_dataset_items(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        limit,
        limit_out_of_range,
        offset,
        offset_out_of_range,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test listing dataset items."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=limit,
                offset=offset,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        assert (
            len(dataset_items) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(db_dataset_items)
        )

    def test_get_dataset_item_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        fetched_dataset_item = fxt_dataset_service.get_dataset_item_by_id(
            project_id=project.id, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert (
            str(fetched_dataset_item.id) == db_dataset_items[0].id
            and fetched_dataset_item.name == db_dataset_items[0].name
        )

    def test_get_dataset_item_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_binary_path_by_id(
        self,
        tmp_path: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item binary path by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_binary_path_by_id(
            project_id=project.id, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert (
            dataset_item_binary_path
            == tmp_path / f"projects/{str(project.id)}/dataset/{db_dataset_items[0].id}.{db_dataset_items[0].format}"
        )

    def test_get_dataset_item_binary_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item binary path raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=project.id, dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_thumbnail_path_by_id(
        self,
        tmp_path: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item thumbnail path by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
            project=project, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert (
            dataset_item_binary_path
            == tmp_path / f"projects/{str(project.id)}/dataset/{db_dataset_items[0].id}-thumb.jpg"
        )

    def test_get_dataset_item_thumbnail_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item thumbnail path raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(project=project, dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        fxt_projects_dir: Path,
        db_session: Session,
    ):
        """Test deleting a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_dir = fxt_projects_dir / str(project.id) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        binary_path = dataset_dir / f"{db_dataset_items[0].id}.{db_dataset_items[0].format}"
        binary_path.touch()
        assert os.path.exists(binary_path)

        thumbnail_path = dataset_dir / f"{db_dataset_items[0].id}-thumb.jpg"
        thumbnail_path.touch()
        assert os.path.exists(thumbnail_path)

        """Test deleting a dataset item."""
        fxt_dataset_service.delete_dataset_item(project=project, dataset_item_id=UUID(db_dataset_items[0].id))

        assert db_session.get(DatasetItemDB, db_dataset_items[0].id) is None
        assert not os.path.exists(binary_path)
        assert not os.path.exists(thumbnail_path)

    def test_delete_dataset_item_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test deleting a non-existent dataset item raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items

        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(project=project, dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    @pytest.mark.parametrize(
        "item_idx, label_idx",
        [
            (0, 0),  # Set annotation with new label to unannotated item
            (1, 0),  # Set annotation with existing label on already annotated item
            (1, 1),  # Set annotation with new label to already annotated item
        ],
    )
    def test_set_dataset_item_annotations(
        self,
        item_idx: int,
        label_idx: int,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
        db_session: Session,
    ):
        """Test setting a dataset item annotation."""
        project, db_dataset_items = fxt_project_with_dataset_items
        label_id = str(project.task.labels[label_idx].id)
        dataset_item_id = db_dataset_items[item_idx].id
        annotations = fxt_annotations(project.task.labels[label_idx].id)
        fxt_dataset_service.set_dataset_item_annotations(
            project=project,
            dataset_item_id=UUID(dataset_item_id),
            annotations=annotations,
        )

        dataset_item = db_session.get(DatasetItemDB, dataset_item_id)
        assert dataset_item is not None
        assert dataset_item.annotation_data is not None
        assert [
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        ] == annotations
        assert (
            db_session.scalar(
                select(func.count())
                .select_from(DatasetItemLabelDB)
                .where(DatasetItemLabelDB.dataset_item_id == dataset_item_id)
            )
            == 1
        )
        assert db_session.get(DatasetItemLabelDB, (dataset_item_id, label_id)) is not None

    def test_set_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
    ):
        """Test setting a dataset item annotation for a non-existent dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()
        annotations = fxt_annotations(project.task.labels[0].id)

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project=project,
                dataset_item_id=non_existent_id,
                annotations=annotations,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ):
        """Test deleting a dataset item annotation."""
        project, db_dataset_items = fxt_project_with_dataset_items
        item_label_id = (db_dataset_items[1].id, str(project.task.labels[0].id))
        assert db_session.get(DatasetItemLabelDB, item_label_id) is not None

        fxt_dataset_service.delete_dataset_item_annotations(
            project=project,
            dataset_item_id=UUID(db_dataset_items[1].id),
        )

        dataset_item = db_session.get(DatasetItemDB, db_dataset_items[1].id)
        assert dataset_item is not None
        assert dataset_item.annotation_data is None
        assert db_session.get(DatasetItemLabelDB, item_label_id) is None

    def test_delete_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test deleting a dataset item annotation."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item_annotations(
                project=project,
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_assign_dataset_item_subset_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.assign_dataset_item_subset(
                project_id=project.id,
                dataset_item_id=non_existent_id,
                subset=DatasetItemSubset.TRAINING,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    @pytest.mark.parametrize(
        "subset", [DatasetItemSubset.TRAINING, DatasetItemSubset.TESTING, DatasetItemSubset.VALIDATION]
    )
    def test_assign_dataset_item_subset_already_assigned(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        subset,
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items

        with pytest.raises(SubsetAlreadyAssignedError):
            fxt_dataset_service.assign_dataset_item_subset(
                project_id=project.id,
                dataset_item_id=UUID(db_dataset_items[2].id),
                subset=subset,
            )

    @pytest.mark.parametrize(
        "subset", [DatasetItemSubset.TRAINING, DatasetItemSubset.TESTING, DatasetItemSubset.VALIDATION]
    )
    def test_assign_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        subset,
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items

        returned_dataset_item = fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id,
            dataset_item_id=UUID(db_dataset_items[0].id),
            subset=subset,
        )

        assert str(returned_dataset_item.id) == db_dataset_items[0].id and returned_dataset_item.subset == subset

    @pytest.mark.parametrize(
        "annotation_status, expected_count",
        [
            (None, 7),  # All items
            ("unannotated", 2),  # 2 unannotated items
            ("reviewed", 3),  # 3 reviewed items
            ("to_review", 2),  # 2 to_review items
        ],
    )
    def test_count_dataset_items_with_annotation_status(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: str | None,
        expected_count: int,
    ) -> None:
        """Test counting dataset items with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        count = fxt_dataset_service.count_dataset_items(project=project, annotation_status=annotation_status)

        assert count == expected_count

    @pytest.mark.parametrize(
        "annotation_status, expected_names",
        [
            (None, ["unannotated1", "unannotated2", "reviewed1", "reviewed2", "reviewed3", "to_review1", "to_review2"]),
            (DatasetItemAnnotationStatus.UNANNOTATED, ["unannotated1", "unannotated2"]),
            (DatasetItemAnnotationStatus.REVIEWED, ["reviewed1", "reviewed2", "reviewed3"]),
            (DatasetItemAnnotationStatus.TO_REVIEW, ["to_review1", "to_review2"]),
        ],
    )
    def test_list_dataset_items_with_annotation_status(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_names: list[str],
    ) -> None:
        """Test listing dataset items with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                annotation_status=annotation_status,
            ),
        )

        assert len(dataset_items) == len(expected_names)
        actual_names = sorted([item.name for item in dataset_items])
        assert actual_names == sorted(expected_names)

    @pytest.mark.parametrize(
        "annotation_status, limit, offset, expected_count",
        [
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 0, 1),  # First page of unannotated
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 1, 1),  # Second page of unannotated
            (DatasetItemAnnotationStatus.UNANNOTATED, 1, 2, 0),  # Beyond available unannotated items
            (DatasetItemAnnotationStatus.REVIEWED, 2, 0, 2),  # First page of reviewed
            (DatasetItemAnnotationStatus.REVIEWED, 2, 2, 1),  # Second page of reviewed (only 1 left)
            (DatasetItemAnnotationStatus.TO_REVIEW, 10, 0, 2),  # All to_review items
        ],
    )
    def test_list_dataset_items_with_annotation_status_pagination(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing dataset items with annotation_status filter and pagination."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=limit,
                offset=offset,
                annotation_status=annotation_status,
            ),
        )

        assert len(dataset_items) == expected_count

    def test_list_dataset_items_annotation_status_combined_with_dates(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test annotation_status filter combined with date filters."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        # All reviewed items within date range
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                start_date=datetime.fromisoformat("2025-01-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-02-02T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.REVIEWED,
            ),
        )
        assert len(dataset_items) == 3
        assert all(item.user_reviewed for item in dataset_items)
        assert all(item.annotation_data is not None for item in dataset_items)

        # No items outside date range
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                start_date=datetime.fromisoformat("2025-03-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-03-31T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.UNANNOTATED,
            ),
        )
        assert len(dataset_items) == 0

    def test_annotation_status_filter_verifies_data_correctness(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that annotation_status filter returns items with correct properties."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        # Unannotated items should have no annotation_data
        unannotated_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                annotation_status=DatasetItemAnnotationStatus.UNANNOTATED,
            ),
        )
        assert len(unannotated_items) == 2
        for item in unannotated_items:
            assert item.annotation_data is None

        # Reviewed items should have annotation_data and user_reviewed=True
        reviewed_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                annotation_status=DatasetItemAnnotationStatus.REVIEWED,
            ),
        )
        assert len(reviewed_items) == 3
        for item in reviewed_items:
            assert item.annotation_data is not None
            assert item.user_reviewed is True

        # To review items should have annotation_data and user_reviewed=False
        to_review_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                annotation_status=DatasetItemAnnotationStatus.TO_REVIEW,
            ),
        )
        assert len(to_review_items) == 2
        for item in to_review_items:
            assert item.annotation_data is not None
            assert item.user_reviewed is False

    def test_list_dataset_items_filter_by_single_label(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing dataset items filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Filter by label_0 - should return items 1 and 3 (item_label_0 and item_both_labels)
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                label_ids=[label_0_id],
            ),
        )

        assert len(dataset_items) == 2
        item_names = {item.name for item in dataset_items}
        assert item_names == {"item_label_0", "item_both_labels"}

    def test_list_dataset_items_filter_by_multiple_labels(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing dataset items filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Filter by label_0 OR label_1 - should return items 1, 2, and 3
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                label_ids=[label_0_id, label_1_id],
            ),
        )

        assert len(dataset_items) == 3
        item_names = {item.name for item in dataset_items}
        assert item_names == {"item_label_0", "item_label_1", "item_both_labels"}

    def test_list_dataset_items_filter_by_nonexistent_label(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing dataset items filtered by a nonexistent label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        nonexistent_label_id = uuid4()

        # Filter by nonexistent label - should return empty list
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                label_ids=[nonexistent_label_id],
            ),
        )

        assert len(dataset_items) == 0

    def test_count_dataset_items_filter_by_single_label(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting dataset items filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Count items with label_0 - should return 2
        count = fxt_dataset_service.count_dataset_items(
            project=project,
            label_ids=[label_0_id],
        )

        assert count == 2

    def test_count_dataset_items_filter_by_multiple_labels(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting dataset items filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Count items with label_0 OR label_1 - should return 3
        count = fxt_dataset_service.count_dataset_items(
            project=project,
            label_ids=[label_0_id, label_1_id],
        )

        assert count == 3

    def test_list_dataset_items_no_label_filter(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing dataset items without label filter returns all items."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items

        # No filter - should return all 4 items
        dataset_items = fxt_dataset_service.list_dataset_items(project_id=project.id)

        assert len(dataset_items) == 4
        item_names = {item.name for item in dataset_items}
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
    def test_count_dataset_items_with_subset(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        expected_count: int,
    ) -> None:
        """Test counting dataset items with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        count = fxt_dataset_service.count_dataset_items(project=project, subset=subset)

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
    def test_list_dataset_items_with_subset(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        expected_names: list[str],
    ) -> None:
        """Test listing dataset items with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=20,
                offset=0,
                subset=subset,
            ),
        )

        assert len(dataset_items) == len(expected_names)
        actual_names = sorted([item.name for item in dataset_items])
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
    def test_list_dataset_items_with_subset_pagination(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: str | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing dataset items with subset filter and pagination."""
        project, db_dataset_items = fxt_project_with_subset_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=limit,
                offset=offset,
                subset=subset,
            ),
        )

        assert len(dataset_items) == expected_count

    def test_subset_filter_verifies_data_correctness(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that subset filter returns items with correct subset values."""
        project, db_dataset_items = fxt_project_with_subset_items

        # Unassigned items should have subset=unassigned
        unassigned_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=20,
                offset=0,
                subset="unassigned",
            ),
        )
        assert len(unassigned_items) == 2
        for item in unassigned_items:
            assert item.subset == DatasetItemSubset.UNASSIGNED

        # Training items should have subset=training
        training_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=20,
                offset=0,
                subset="training",
            ),
        )
        assert len(training_items) == 3
        for item in training_items:
            assert item.subset == DatasetItemSubset.TRAINING

        # Validation items should have subset=validation
        validation_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=20,
                offset=0,
                subset="validation",
            ),
        )
        assert len(validation_items) == 2
        for item in validation_items:
            assert item.subset == DatasetItemSubset.VALIDATION

        # Testing items should have subset=testing
        testing_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=20,
                offset=0,
                subset="testing",
            ),
        )
        assert len(testing_items) == 1
        for item in testing_items:
            assert item.subset == DatasetItemSubset.TESTING

    def test_save_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test saving a dataset revision."""
        project, db_dataset_items = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(project.id, project.task, DatasetItemAnnotationStatus.REVIEWED)

        revision_id = fxt_dataset_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Verify that a revision entry was created
        assert db_session.get(DatasetRevisionDB, str(revision_id)) is not None
        assert (fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id) / "dataset.zip").exists()
