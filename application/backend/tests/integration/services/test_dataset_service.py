# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB, PipelineDB
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    Image,
    LabelReference,
    Pipeline,
    Project,
    Rectangle,
    Video,
    VideoFrame,
)
from app.models.media import MediaAdapter
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_service import DatasetItemFilters, DatasetService, SubsetAlreadyAssignedError
from app.services.media_service import MediaService


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

    configs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (
            {"type": "image", "name": "test1", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
            {"subset": "unassigned"},
        ),
        (
            {
                "type": "image",
                "name": "test2",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [
                    {"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}
                ],
            },
        ),
        (
            {"type": "image", "name": "test3", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
            {"subset": "training"},
        ),
    ]

    db_dataset_items = []
    for media_config, dataset_item_config in configs:
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

    db_dataset_items = []
    for list in [unannotated_items, reviewed_items]:
        for idx, dataset_item in enumerate(list):
            db_media = MediaDB(
                type="image",
                name=f"{dataset_item.subset}{idx + 1}",
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
    for item in [*reviewed_items]:
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
    db_dataset_items = [*unassigned_items, *training_items, *validation_items, *testing_items]
    for idx, dataset_item in enumerate(db_dataset_items):
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

    db_session.add_all(db_dataset_items)
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_rich_dataset_items(fxt_project_with_pipeline, db_session) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with images, videos, annotated video frames, and a dataset item with multiple annotations."""
    project, _ = fxt_project_with_pipeline

    label_1_id = str(project.task.labels[0].id)
    label_2_id = str(project.task.labels[1].id)

    db_media_items = []
    db_frame_items = []
    db_dataset_items = []

    default_media_values = {
        "size": 1024,
        "width": 1024,
        "height": 768,
        "project_id": str(project.id),
        "created_at": datetime.fromisoformat("2025-02-01T00:00:00Z"),
    }
    default_item_values = {
        "project_id": str(project.id),
        "subset": "training",
        "created_at": datetime.fromisoformat("2025-02-01T00:00:00Z"),
    }
    annotation_data_1 = {
        "labels": [{"id": label_1_id}],
        "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
    }
    annotation_data_2 = {
        "labels": [{"id": label_2_id}],
        "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
    }

    # Image (unannotated)
    unannotated_image = MediaDB(
        id=str(uuid4()),
        type="image",
        name="img1",
        format="jpg",
        **default_media_values,
    )
    item_unannotated_image = DatasetItemDB(
        id=str(unannotated_image.id),
        **default_item_values,
    )
    db_media_items.append(unannotated_image)
    db_dataset_items.append(item_unannotated_image)

    # Image (annotated with multiple annotations (2 with label_1, 1 with label_2)
    annotated_image = MediaDB(
        id=str(uuid4()),
        type="image",
        name="multi_ann",
        format="jpg",
        **default_media_values,
    )
    item_annotated_image = DatasetItemDB(
        id=str(annotated_image.id),
        **default_item_values,
        annotation_data=[annotation_data_1, annotation_data_1, annotation_data_2],
        user_reviewed=True,
    )
    db_media_items.append(annotated_image)
    db_dataset_items.append(item_annotated_image)

    # Image (annotated with no objects - empty annotations)
    no_object_image = MediaDB(
        id=str(uuid4()),
        type="image",
        name="no_object_img",
        format="jpg",
        **default_media_values,
    )
    item_no_object_image = DatasetItemDB(
        id=str(no_object_image.id),
        **default_item_values,
        annotation_data=[],
        user_reviewed=True,
    )
    db_media_items.append(no_object_image)
    db_dataset_items.append(item_no_object_image)

    # Video (annotated frames)
    annotated_video = MediaDB(
        id=str(uuid4()),
        type="video",
        name="vid1",
        format="mp4",
        fps=30.0,
        frame_count=20,
        **default_media_values,
    )
    db_media_items.append(annotated_video)

    # Video frame (annotated with label_1)
    annotated_video_frame_1 = MediaDB(
        id=str(uuid4()),
        type="video_frame",
        name="vf1",
        format="jpg",
        video_id=str(annotated_video.id),
        frame_index=3,
        **default_media_values,
    )
    item_annotated_video_frame_1 = DatasetItemDB(
        id=str(annotated_video_frame_1.id),
        **default_item_values,
        annotation_data=[annotation_data_1],
        user_reviewed=True,
    )
    db_frame_items.append(annotated_video_frame_1)
    db_dataset_items.append(item_annotated_video_frame_1)

    # Video frame (annotated with label_2)
    annotated_video_frame_2 = MediaDB(
        id=str(uuid4()),
        type="video_frame",
        name="vf2",
        format="jpg",
        video_id=str(annotated_video.id),
        frame_index=8,
        **default_media_values,
    )
    item_annotated_video_frame_2 = DatasetItemDB(
        id=str(annotated_video_frame_2.id),
        **default_item_values,
        annotation_data=[annotation_data_2],
        user_reviewed=True,
    )
    db_frame_items.append(annotated_video_frame_2)
    db_dataset_items.append(item_annotated_video_frame_2)

    # Video (no annotated frames)
    unannotated_video = MediaDB(
        id=str(uuid4()),
        type="video",
        name="vid2",
        format="mp4",
        fps=30.0,
        frame_count=15,
        **default_media_values,
    )
    db_media_items.append(unannotated_video)

    [db_session.add(item) for item in db_media_items]
    db_session.flush()
    db_session.commit()
    [db_session.add(item) for item in db_frame_items]
    db_session.flush()
    [db_session.add(item) for item in db_dataset_items]
    db_session.flush()

    db_session.add(DatasetItemLabelDB(dataset_item_id=item_annotated_video_frame_1.id, label_id=label_1_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=item_annotated_video_frame_2.id, label_id=label_2_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=item_annotated_image.id, label_id=label_1_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=item_annotated_image.id, label_id=label_2_id))
    db_session.flush()

    return project, db_dataset_items


class TestDatasetServiceIntegration:
    """Integration tests for DatasetService."""

    @pytest.mark.parametrize("use_pipeline_model", [True, False])
    @pytest.mark.parametrize("user_reviewed", [True, False])
    def test_create_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
        db_session: Session,
        user_reviewed: bool,
        use_pipeline_model: bool,
    ) -> None:
        """Test creating a dataset item."""
        project, pipeline = fxt_project_with_pipeline
        label_id = project.task.labels[0].id

        db_media = MediaDB(
            type="image",
            name="test",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
        )
        db_session.add(db_media)
        db_session.flush()
        media = MediaAdapter.validate_python(db_media, from_attributes=True)

        created_dataset_item = fxt_dataset_service.create_dataset_item(
            project_id=project.id,
            task=project.task,
            media=media,
            user_reviewed=user_reviewed,
            prediction_model_id=pipeline.model_id if use_pipeline_model else None,
            annotations=fxt_annotations(label_id) if not user_reviewed else None,
        )

        dataset_item = db_session.get(DatasetItemDB, str(created_dataset_item.id))
        assert dataset_item is not None
        assert (
            dataset_item.id == str(created_dataset_item.id)
            and dataset_item.project_id == str(project.id)
            and dataset_item.id == str(media.id)
            and dataset_item.user_reviewed == user_reviewed
            and dataset_item.subset == DatasetItemSubset.UNASSIGNED
            and dataset_item.subset_assigned_at is None
        )
        if use_pipeline_model:
            assert dataset_item.prediction_model_id == str(pipeline.model_id)
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
    def test_list_dataset_items_with_media(
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
        """Test listing dataset items with corresponding media."""
        project, db_dataset_items = fxt_project_with_dataset_items

        list = fxt_dataset_service.list_dataset_items_with_media(
            project_id=project.id,
            filters=DatasetItemFilters(
                limit=limit,
                offset=offset,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        assert (
            len(list) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(db_dataset_items)
        )
        for dataset_item, media in list:
            assert isinstance(dataset_item, DatasetItem) and isinstance(media, Image | Video | VideoFrame)

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

        assert str(fetched_dataset_item.id) == db_dataset_items[0].id

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

    @pytest.mark.parametrize(
        "item_idx, label_idx",
        [
            (0, 0),  # Set annotation with new label to unannotated item
            (2, 0),  # Set annotation with existing label on already reviewed item
            (2, 1),  # Set annotation with new label to already reviewed item
        ],
    )
    def test_set_dataset_item_annotations(
        self,
        item_idx: int,
        label_idx: int,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
        db_session: Session,
    ):
        """Test setting a dataset item annotation."""
        _, pipeline = fxt_project_with_pipeline
        project, db_dataset_items = fxt_project_with_annotation_status_items
        label_id = str(project.task.labels[label_idx].id)
        dataset_item_id = db_dataset_items[item_idx].id
        annotations = fxt_annotations(project.task.labels[label_idx].id)
        fxt_dataset_service.set_dataset_item_annotations(
            project=project,
            dataset_item_id=UUID(dataset_item_id),
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=pipeline.model_id,
        )

        dataset_item = db_session.get(DatasetItemDB, dataset_item_id)
        assert dataset_item is not None
        assert dataset_item.annotation_data is not None
        assert dataset_item.user_reviewed is True
        assert dataset_item.prediction_model_id == str(pipeline.model_id)
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
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        fxt_annotations: Callable[[UUID], list[DatasetItemAnnotation]],
        db_session: Session,
    ):
        """Test setting a dataset item annotation for a non-existent dataset item."""
        _, pipeline = fxt_project_with_pipeline
        project, db_dataset_items = fxt_project_with_dataset_items
        annotations = fxt_annotations(project.task.labels[0].id)

        # Create a media record with no corresponding dataset item so the media
        # lookup succeeds but the dataset item update fails.
        orphan_media = MediaDB(
            type="image",
            name="orphan",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
        )
        db_session.add(orphan_media)
        db_session.flush()
        orphan_media_id = UUID(orphan_media.id)

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project=project,
                dataset_item_id=orphan_media_id,
                annotations=annotations,
                user_reviewed=True,
                prediction_model_id=pipeline.model_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(orphan_media_id)

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

    @pytest.mark.parametrize("subset", [DatasetItemSubset.TESTING, DatasetItemSubset.VALIDATION])
    def test_assign_dataset_item_subset_already_assigned(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
        subset,
    ):
        """Test assigning a different subset to a dataset item that already has one raises an error."""
        project, db_dataset_items = fxt_project_with_dataset_items

        with pytest.raises(SubsetAlreadyAssignedError):
            fxt_dataset_service.assign_dataset_item_subset(
                project_id=project.id,
                dataset_item_id=UUID(db_dataset_items[2].id),
                subset=subset,
            )

    def test_assign_dataset_item_subset_same_subset_is_noop(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test assigning the same subset that is already assigned does not raise an error."""
        project, db_dataset_items = fxt_project_with_dataset_items

        # db_dataset_items[2] already has "training" subset
        returned_dataset_item = fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id,
            dataset_item_id=UUID(db_dataset_items[2].id),
            subset=DatasetItemSubset.TRAINING,
        )

        assert str(returned_dataset_item.id) == db_dataset_items[2].id
        assert returned_dataset_item.subset == DatasetItemSubset.TRAINING

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
            (None, 5),  # All items
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 2),  # 2 items without annotations
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 3),  # 3 items with annotations
        ],
    )
    def test_count_dataset_items_with_annotation_status(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_count: int,
    ) -> None:
        """Test counting dataset items with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        count = fxt_dataset_service.count_dataset_items(project=project, annotation_status=annotation_status)

        assert count == expected_count

    @pytest.mark.parametrize(
        "annotation_status, expected_names",
        [
            (None, ["unannotated1", "unannotated2", "reviewed1", "reviewed2", "reviewed3"]),
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, ["unannotated1", "unannotated2"]),
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, ["reviewed1", "reviewed2", "reviewed3"]),
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

    @pytest.mark.parametrize(
        "annotation_status, limit, offset, expected_count",
        [
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 0, 1),  # First page of unannotated
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 1, 1),  # Second page of unannotated
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 2, 0),  # Beyond available unannotated items
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 2, 0, 2),  # First page of reviewed
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 2, 2, 1),  # Second page of reviewed (only 1 left)
            (None, 10, 0, 5),  # All items
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
                annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
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
                annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
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
                annotation_status=DatasetItemAnnotationStatus.MISSING_ANNOTATIONS,
            ),
        )
        assert len(unannotated_items) == 2
        for item in unannotated_items:
            assert item.annotation_data is None

        # Reviewed items should have user_reviewed=True
        reviewed_items = fxt_dataset_service.list_dataset_items(
            project_id=project.id,
            filters=DatasetItemFilters(
                annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            ),
        )
        assert len(reviewed_items) == 3
        for item in reviewed_items:
            assert item.user_reviewed is True

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

    def test_get_dataset_statistics(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_rich_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test retrieving dataset statistics with images, videos, video frames, and multiple annotation instances."""
        project, _ = fxt_project_with_rich_dataset_items

        statistics = fxt_dataset_service.get_dataset_statistics(project_id=project.id)

        # Media counts
        assert statistics.media_counts.images == 3
        assert statistics.media_counts.videos == 2
        assert statistics.media_counts.video_frames == 35

        # Annotation counts
        assert statistics.annotations_counts.annotated_images == 2
        assert statistics.annotations_counts.annotated_videos == 1
        assert statistics.annotations_counts.annotated_video_frames == 2

        # Instances counts
        assert statistics.annotations_counts.instances == 5
        assert len(statistics.annotations_counts.instances_per_label) == 3
        label_counts = {
            str(lbl.label_id) if lbl.label_id is not None else None: lbl.instances
            for lbl in statistics.annotations_counts.instances_per_label
        }
        assert label_counts[str(project.task.labels[0].id)] == 3
        assert label_counts[str(project.task.labels[1].id)] == 2
        assert label_counts[None] == 1

    def test_get_dataset_statistics_no_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ):
        """Test retrieving dataset statistics when no media have annotations (annotation_data is None).

        This is a regression test for a bug where get_statistics() would raise
        TypeError: 'NoneType' object is not iterable when iterating over items
        with None annotation_data.
        """
        project, _ = fxt_project_with_pipeline

        # Add images with no annotations
        for i in range(3):
            db_media = MediaDB(
                type="image",
                name=f"unannotated_{i}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
            )
            db_session.add(db_media)
            db_session.flush()
            db_session.add(
                DatasetItemDB(
                    id=str(db_media.id),
                    project_id=str(project.id),
                    subset="training",
                    user_reviewed=False,
                )
            )
        db_session.flush()

        # This should not raise TypeError
        statistics = fxt_dataset_service.get_dataset_statistics(project_id=project.id)

        assert statistics.media_counts.images == 3
        assert statistics.media_counts.videos == 0
        assert statistics.media_counts.video_frames == 0
        assert statistics.annotations_counts.annotated_images == 0
        assert statistics.annotations_counts.instances == 0
        assert len(statistics.annotations_counts.instances_per_label) == 1
        assert statistics.annotations_counts.instances_per_label[0].label_id is None
        assert statistics.annotations_counts.instances_per_label[0].instances == 0

    def test_get_dataset_statistics_mixed_annotated_and_unannotated(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ):
        """Test dataset statistics with a mix of annotated and unannotated items.

        Ensures items with None annotation_data are safely skipped while annotated items
        are correctly counted.
        """
        project, _ = fxt_project_with_pipeline
        label_id = str(project.task.labels[0].id)

        annotation_data = [
            {
                "labels": [{"id": label_id}],
                "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
            }
        ]

        # Add an unannotated image
        unannotated_media = MediaDB(
            type="image",
            name="unannotated",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
        )
        db_session.add(unannotated_media)
        db_session.flush()
        db_session.add(
            DatasetItemDB(
                id=str(unannotated_media.id),
                project_id=str(project.id),
                subset="training",
                user_reviewed=False,
            )
        )

        # Add an annotated image
        annotated_media = MediaDB(
            type="image",
            name="annotated",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
        )
        db_session.add(annotated_media)
        db_session.flush()
        db_session.add(
            DatasetItemDB(
                id=str(annotated_media.id),
                project_id=str(project.id),
                subset="training",
                annotation_data=annotation_data,
                user_reviewed=True,
            )
        )
        db_session.flush()

        db_session.add(DatasetItemLabelDB(dataset_item_id=str(annotated_media.id), label_id=label_id))
        db_session.flush()

        statistics = fxt_dataset_service.get_dataset_statistics(project_id=project.id)

        assert statistics.media_counts.images == 2
        assert statistics.annotations_counts.annotated_images == 1
        assert statistics.annotations_counts.instances == 1
        assert len(statistics.annotations_counts.instances_per_label) == 2
        label_counts = {
            str(lbl.label_id) if lbl.label_id is not None else None: lbl.instances
            for lbl in statistics.annotations_counts.instances_per_label
        }
        assert label_counts[label_id] == 1
        assert label_counts[None] == 0
