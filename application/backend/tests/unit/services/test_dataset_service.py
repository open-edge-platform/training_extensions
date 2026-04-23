# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import re
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import MediaDB
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    FullImage,
    Label,
    LabelReference,
    Point,
    Polygon,
    Project,
    Rectangle,
    Task,
    TaskType,
)
from app.repositories import DatasetItemRepository
from app.services import DatasetService, LabelService, MediaService
from app.services.dataset_service import AnnotationValidationError


class TestDatasetServiceUnit:
    """Unit tests for DatasetService."""

    @pytest.fixture
    def fxt_dataset_service(self):
        db_session = MagicMock(spec=Session)
        label_service = MagicMock(spec=LabelService)
        media_service = MagicMock(spec=MediaService)
        return DatasetService(
            label_service=label_service,
            media_service=media_service,
            db_session=db_session,
        )

    @pytest.fixture
    def fxt_multiclass_classification_project(self):
        return Project(
            id=uuid4(),
            name="Test Multiclass Classification Project",
            task=Task(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
            ),
            active_pipeline=False,
            created_at=datetime.now(tz=UTC),
        )

    @pytest.fixture
    def fxt_multilabel_classification_project(self):
        return Project(
            id=uuid4(),
            name="Test Multilabel Classification Project",
            task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False),
            active_pipeline=False,
            created_at=datetime.now(tz=UTC),
        )

    @pytest.fixture
    def fxt_detection_project(self):
        return Project(
            id=uuid4(),
            name="Test Detection Project",
            task=Task(task_type=TaskType.DETECTION),
            active_pipeline=False,
            created_at=datetime.now(tz=UTC),
        )

    @pytest.fixture
    def fxt_segmentation_project(self):
        return Project(
            id=uuid4(),
            name="Test Instance Segmentation Project",
            task=Task(task_type=TaskType.INSTANCE_SEGMENTATION),
            active_pipeline=False,
            created_at=datetime.now(tz=UTC),
        )

    def test_validate_annotations_labels(self) -> None:
        label_id = uuid4()
        labels = [Label(id=label_id, project_id=uuid4(), name="cat", color="#00FF00", hotkey="c")]
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)

    def test_validate_annotations_labels_not_found(self) -> None:
        label_id = uuid4()
        labels = [Label(id=label_id, project_id=uuid4(), name="cat", color="#00FF00", hotkey="c")]
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)

    def test_validate_annotations_coordinates_rectangle(self) -> None:
        media = MediaDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService._validate_annotations_coordinates(annotations=annotations, media=media)

    @pytest.mark.parametrize(
        "x, y, width, height, validation_msg",
        [
            (1000, 0, 10, 10, "Rectangle coordinates (x1=1000, x2=1010) are out of bounds for media width 100"),
            (0, 1000, 10, 10, "Rectangle coordinates (y1=1000, y2=1010) are out of bounds for media height 50"),
            (0, 0, 1000, 10, "Rectangle coordinates (x1=0, x2=1000) are out of bounds for media width 100"),
            (0, 0, 10, 1000, "Rectangle coordinates (y1=0, y2=1000) are out of bounds for media height 50"),
        ],
    )
    def test_validate_annotations_coordinates_invalid_rectangle(self, x, y, width, height, validation_msg):
        media = MediaDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=x, y=y, width=width, height=height),
            )
        ]
        with pytest.raises(AnnotationValidationError, match=re.escape(validation_msg)):
            DatasetService._validate_annotations_coordinates(annotations=annotations, media=media)

    def test_validate_annotations_coordinates_polygon(self) -> None:
        media = MediaDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            )
        ]
        DatasetService._validate_annotations_coordinates(annotations=annotations, media=media)

    @pytest.mark.parametrize(
        "x, y, validation_msg",
        [
            (1000, 10, "Polygon points (x=1000.0, y=10.0) are out of bounds for media (100, 50)"),
            (10, 1000, "Polygon points (x=10.0, y=1000.0) are out of bounds for media (100, 50)"),
        ],
    )
    def test_validate_annotations_coordinates_invalid_polygon(self, x, y, validation_msg):
        media = MediaDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=x, y=y)]),
            )
        ]
        with pytest.raises(AnnotationValidationError, match=re.escape(validation_msg)):
            DatasetService._validate_annotations_coordinates(annotations=annotations, media=media)

    def test_validate_annotations_multilabel_classification(self, fxt_multilabel_classification_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        # Should not raise any exception since multilabel classification allows multiple labels
        DatasetService._validate_annotation_shapes(
            annotations=annotations, task=fxt_multilabel_classification_project.task
        )

    def test_validate_annotations_multilabel_classification_empty(self, fxt_multilabel_classification_project) -> None:
        # Should not raise any exception since multilabel classification allows the empty label
        DatasetService._validate_annotation_shapes(annotations=[], task=fxt_multilabel_classification_project.task)

    def test_validate_annotations_multilabel_classification_multi_annotations(
        self, fxt_multilabel_classification_project
    ) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            ),
        ]
        # Should raise an exception since multilabel classification does not allow multiple annotations
        # (only one 'full_image' shape, possibly with multiple labels, is allowed)
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(
                annotations=annotations, task=fxt_multilabel_classification_project.task
            )

    @pytest.mark.parametrize(
        "shape",
        [
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_multilabel_classification_wrong_shape(
        self, shape, fxt_multilabel_classification_project
    ):
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=shape,
            )
        ]
        # Should raise an exception since classification only allows 'full_image' shape
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(
                annotations=annotations, task=fxt_multilabel_classification_project.task
            )

    def test_validate_annotations_multiclass_classification(self, fxt_multiclass_classification_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        # Should not raise any exception since the annotation is valid for multiclass classification
        DatasetService._validate_annotation_shapes(
            annotations=annotations, task=fxt_multiclass_classification_project.task
        )

    def test_validate_annotations_multiclass_classification_empty(self, fxt_multiclass_classification_project) -> None:
        # Should raise an exception since multiclass classification requires exactly one label (empty label not allowed)
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(annotations=[], task=fxt_multiclass_classification_project.task)

    def test_validate_annotations_multiclass_classification_multiple_labels(
        self, fxt_multiclass_classification_project
    ) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        # Should raise an exception since multiclass classification does not allow multiple labels
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(
                annotations=annotations, task=fxt_multiclass_classification_project.task
            )

    def test_validate_annotations_detection(self, fxt_detection_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=10, y=10, width=10, height=10),
            ),
        ]
        # Should not raise any exception since the annotations are valid for detection task
        DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_detection_project.task)

    def test_validate_annotations_detection_empty(self, fxt_detection_project) -> None:
        # Should not raise any exception since detection task allows the empty label
        DatasetService._validate_annotation_shapes(annotations=[], task=fxt_detection_project.task)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_detection_wrong_shape(self, shape, fxt_detection_project):
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
        ]
        # Should raise an exception since detection task only allows 'rectangle' shape
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_detection_project.task)

    def test_validate_annotations_detection_wrong_shape_multiple_labels(self, fxt_detection_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=10, y=10, width=10, height=10),
            ),
        ]
        # Should raise an exception since detection task does not allow multiple labels on the same shape
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_detection_project.task)

    def test_validate_annotations_segmentation(self, fxt_segmentation_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=10, y=10), Point(x=20, y=20)]),
            ),
        ]
        # Should not raise any exception since the annotations are valid for segmentation task
        DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_segmentation_project.task)

    def test_validate_annotations_segmentation_empty(self, fxt_segmentation_project) -> None:
        # Should not raise any exception since segmentation task allows the empty label
        DatasetService._validate_annotation_shapes(annotations=[], task=fxt_segmentation_project.task)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
        ],
    )
    def test_validate_annotations_segmentation_wrong_shape(self, shape, fxt_segmentation_project):
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
        ]
        # Should raise an exception since segmentation task only allows 'polygon' shape
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_segmentation_project.task)

    def test_validate_annotations_segmentation_wrong_shape_multiple_labels(self, fxt_segmentation_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=10, y=10), Point(x=20, y=20)]),
            ),
        ]
        # Should raise an exception since segmentation task does not allow multiple labels on the same shape
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotation_shapes(annotations=annotations, task=fxt_segmentation_project.task)

    def test_set_dataset_item_annotations(self, fxt_dataset_service, fxt_detection_project) -> None:
        dataset_service = fxt_dataset_service
        dataset_item_id = uuid4()
        dataset_item = MagicMock(spec=DatasetItem)
        label_id = uuid4()
        dataset_item_annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]

        with (
            patch.object(
                DatasetService,
                "_cleanup_and_validate_annotations",
                side_effect=lambda annotations, **kwargs: annotations,
            ) as mock_cleanup_and_validate,
            patch.object(DatasetService, "get_dataset_item_by_id", return_value=dataset_item),
            patch.object(DatasetItemRepository, "set_annotation_data") as mock_repo_set_annotation_data,
            patch.object(DatasetItemRepository, "set_labels") as mock_repo_set_labels,
        ):
            result = dataset_service.set_dataset_item_annotations(
                project=fxt_detection_project,
                dataset_item_id=dataset_item_id,
                annotations=dataset_item_annotations,
                user_reviewed=True,
                prediction_model_id=None,
            )

        mock_cleanup_and_validate.assert_called_once()
        mock_repo_set_annotation_data.assert_called_once_with(
            obj_id=str(dataset_item_id),
            annotation_data=[
                {
                    "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    "labels": [{"id": str(label_id)}],
                    "confidences": None,
                }
            ],
            user_reviewed=True,
            prediction_model_id=None,
        )
        mock_repo_set_labels.assert_called_once_with(
            dataset_item_id=str(dataset_item_id),
            label_ids={str(label_id)},
        )
        assert result == dataset_item

    @pytest.mark.parametrize("user_reviewed", [True, False])
    def test_set_annotations_confidences_handling(
        self, user_reviewed, fxt_dataset_service, fxt_detection_project
    ) -> None:
        """When user_reviewed=True confidences are stripped; when False they are preserved."""
        dataset_service = fxt_dataset_service
        dataset_item_id = uuid4()
        dataset_item = MagicMock(spec=DatasetItem)
        label_id = uuid4()
        prediction_model_id = None if user_reviewed else uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                confidences=[0.95],
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=20, y=20, width=30, height=30),
                confidences=None,
            ),
        ]

        with (
            patch.object(DatasetService, "_validate_annotations_labels"),
            patch.object(DatasetService, "_validate_annotation_shapes"),
            patch.object(DatasetService, "_validate_annotations_coordinates"),
            patch.object(DatasetService, "get_dataset_item_by_id", return_value=dataset_item),
            patch.object(DatasetItemRepository, "set_annotation_data") as mock_repo_set_annotation_data,
            patch.object(DatasetItemRepository, "set_labels"),
        ):
            dataset_service.set_dataset_item_annotations(
                project=fxt_detection_project,
                dataset_item_id=dataset_item_id,
                annotations=annotations,
                user_reviewed=user_reviewed,
                prediction_model_id=prediction_model_id,
            )

        saved_annotation_data = mock_repo_set_annotation_data.call_args.kwargs["annotation_data"]
        if user_reviewed:
            assert all(ann["confidences"] is None for ann in saved_annotation_data)
        else:
            assert saved_annotation_data[0]["confidences"] == [0.95]
            assert saved_annotation_data[1]["confidences"] is None
