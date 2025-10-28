# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
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

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, PipelineDB
from app.schemas import PipelineView, ProjectView
from app.schemas.dataset_item import DatasetItemAnnotation, DatasetItemAnnotationsWithSource, DatasetItemSubset
from app.schemas.label import LabelReference
from app.schemas.shape import FullImage, Rectangle
from app.services import LabelService, PipelineService, ProjectService
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_service import (
    DatasetService,
    InvalidImageError,
    NotAnnotatedError,
    SubsetAlreadyAssignedError,
)
from app.services.event.event_bus import EventBus

logger = logging.getLogger(__name__)


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
    fxt_projects_dir: Path, db_session: Session, fxt_project_service: ProjectService, fxt_label_service: LabelService
) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(fxt_projects_dir.parent, db_session=db_session)


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
) -> tuple[ProjectView, PipelineView]:
    """Fixture to create a ProjectView."""

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
def fxt_project_with_dataset_items(fxt_project_with_pipeline, db_session) -> tuple[ProjectView, list[DatasetItemDB]]:
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
def fxt_annotations() -> Callable[[UUID], list[DatasetItemAnnotation]]:
    def _create_annotations(label_id: UUID) -> list[DatasetItemAnnotation]:
        return [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]

    return _create_annotations


class TestDatasetServiceIntegration:
    """Integration tests for DatasetService."""

    @pytest.mark.parametrize("use_pipeline_model", [True, False])
    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("user_reviewed", [True, False])
    @pytest.mark.parametrize("format", ["jpg", "png"])
    def test_create_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[ProjectView, PipelineView],
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
            prediction_model_id=pipeline.model_id if use_pipeline_model else None,
            annotations=fxt_annotations(label_id) if not user_reviewed else None,
        )
        logger.info(f"Created dataset item: {created_dataset_item}")

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
            assert dataset_item.prediction_model_id == str(pipeline.model_id)
        else:
            assert dataset_item.prediction_model_id is None
        if not user_reviewed:
            assert dataset_item.annotation_data == [
                {
                    "labels": [{"id": str(label_id)}],
                    "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    "confidence": None,
                }
            ]
            assert db_session.get(DatasetItemLabelDB, (str(created_dataset_item.id), str(label_id))) is not None
        else:
            assert dataset_item.annotation_data is None

        binary_file_path = Path(f"data/projects/{project.id}/dataset/{created_dataset_item.id}.{format}")
        assert os.path.exists(binary_file_path)
        assert created_dataset_item.size == os.path.getsize(binary_file_path)

        thumbnail_file_path = Path(f"data/projects/{project.id}/dataset/{created_dataset_item.id}-thumb.jpg")
        assert os.path.exists(thumbnail_file_path)

    def test_create_dataset_item_invalid_image(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[ProjectView, PipelineView],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
            project=project,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )

        assert (
            len(dataset_items) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(db_dataset_items)
        )

    def test_get_dataset_item_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        fetched_dataset_item = fxt_dataset_service.get_dataset_item_by_id(
            project=project, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert (
            str(fetched_dataset_item.id) == db_dataset_items[0].id
            and fetched_dataset_item.name == db_dataset_items[0].name
        )

    def test_get_dataset_item_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(project=project, dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_binary_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item binary path by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_binary_path_by_id(
            project=project, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert dataset_item_binary_path == Path(
            f"data/projects/{str(project.id)}/dataset/{db_dataset_items[0].id}.{db_dataset_items[0].format}"
        )

    def test_get_dataset_item_binary_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item binary path raises error."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(project=project, dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_thumbnail_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item thumbnail path by ID."""
        project, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
            project=project, dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert dataset_item_binary_path == Path(
            f"data/projects/{str(project.id)}/dataset/{db_dataset_items[0].id}-thumb.jpg"
        )

    def test_get_dataset_item_thumbnail_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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

    def test_get_dataset_item_annotations_none(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation and it's missing."""
        project, db_dataset_items = fxt_project_with_dataset_items

        with pytest.raises(NotAnnotatedError):
            fxt_dataset_service.get_dataset_item_annotations(
                project=project,
                dataset_item_id=UUID(db_dataset_items[0].id),
            )

    def test_get_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation."""
        project, db_dataset_items = fxt_project_with_dataset_items

        annotations = fxt_dataset_service.get_dataset_item_annotations(
            project=project,
            dataset_item_id=UUID(db_dataset_items[1].id),
        )

        assert annotations == DatasetItemAnnotationsWithSource(
            annotations=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=project.task.labels[0].id)],
                    shape=FullImage(type="full_image"),
                )
            ],
            user_reviewed=False,
            prediction_model_id=None,
        )

    def test_get_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_annotations(
                project=project,
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.assign_dataset_item_subset(
                project=project,
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
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
        subset,
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items

        with pytest.raises(SubsetAlreadyAssignedError):
            fxt_dataset_service.assign_dataset_item_subset(
                project=project,
                dataset_item_id=UUID(db_dataset_items[2].id),
                subset=subset,
            )

    @pytest.mark.parametrize(
        "subset", [DatasetItemSubset.TRAINING, DatasetItemSubset.TESTING, DatasetItemSubset.VALIDATION]
    )
    def test_assign_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectView, list[DatasetItemDB]],
        subset,
    ):
        """Test assigning a subset to a dataset item."""
        project, db_dataset_items = fxt_project_with_dataset_items

        returned_dataset_item = fxt_dataset_service.assign_dataset_item_subset(
            project=project,
            dataset_item_id=UUID(db_dataset_items[0].id),
            subset=subset,
        )

        assert str(returned_dataset_item.id) == db_dataset_items[0].id and returned_dataset_item.subset == subset
