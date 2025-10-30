# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.components.annotation import AnnotationType, Bbox, Ellipse, Label, Points, Polygon
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem

from otx.data.utils.pre_filtering import is_valid_anno_for_task, pre_filtering
from otx.types.task import OTXTaskType


@pytest.fixture()
def fxt_dm_dataset_with_unannotated() -> DmDataset:
    dataset_items = [
        DatasetItem(
            id=f"item00{i}_non_empty",
            subset="train",
            media=None,
            annotations=[
                Bbox(x=0, y=0, w=1, h=1, label=0),
                Label(label=i % 3),
            ],
        )
        for i in range(1, 81)
    ]
    dataset_items.append(
        DatasetItem(
            id="item000_wrong_bbox",
            subset="train",
            media=None,
            annotations=[
                Bbox(x=0, y=0, w=-1, h=-1, label=0),
                Label(label=0),
            ],
        ),
    )
    dataset_items.append(
        DatasetItem(
            id="item000_wrong_polygon",
            subset="train",
            media=None,
            annotations=[
                Bbox(x=0, y=0, w=-1, h=-1, label=0),
                Polygon(points=[0.1, 0.2, 0.1, 0.2, 0.1, 0.2], label=0),
                Label(label=0),
            ],
        ),
    )
    dataset_items.extend(
        [
            DatasetItem(
                id=f"item00{i}_empty",
                subset="train",
                media=None,
                annotations=[],
            )
            for i in range(20)
        ],
    )
    return DmDataset.from_iterable(dataset_items, categories=["0", "1", "2", "3"])


@pytest.mark.parametrize("unannotated_items_ratio", [0.0, 0.1, 0.5, 1.0])
def test_pre_filtering(fxt_dm_dataset_with_unannotated: DmDataset, unannotated_items_ratio: float) -> None:
    """Test function for pre_filtering.

    Args:
        fxt_dm_dataset_with_unannotated (DmDataset): The dataset to be filtered.
        unannotated_items_ratio (float): The ratio of unannotated background items to be added.

    Returns:
        None
    """
    empty_items = [
        item for item in fxt_dm_dataset_with_unannotated if item.subset == "train" and len(item.annotations) == 0
    ]
    assert len(fxt_dm_dataset_with_unannotated) == 102
    assert len(empty_items) == 20

    filtered_dataset = pre_filtering(
        dataset=fxt_dm_dataset_with_unannotated,
        data_format="datumaro",
        task=OTXTaskType.MULTI_CLASS_CLS,
        unannotated_items_ratio=unannotated_items_ratio,
    )
    assert len(filtered_dataset) == 82 + int(len(empty_items) * unannotated_items_ratio)
    assert len(filtered_dataset.categories()[AnnotationType.label]) == 3


@pytest.fixture()
def fxt_dataset_item() -> DatasetItem:
    """Create a sample dataset item for testing."""
    return DatasetItem(
        id="test_item",
        subset="train",
        media=None,
        annotations=[],
    )


class TestIsValidAnnoForTask:
    """Test cases for is_valid_anno_for_task function."""

    @pytest.mark.parametrize(
        ("task", "annotation", "expected"),
        [
            # DETECTION task tests
            (OTXTaskType.DETECTION, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.DETECTION, Bbox(x=0, y=0, w=-1, h=-1, label=0), False),  # Invalid bbox
            (OTXTaskType.DETECTION, Bbox(x=10, y=10, w=5, h=5, label=0), True),
            (OTXTaskType.DETECTION, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), False),  # Wrong type
            (OTXTaskType.DETECTION, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), False),
            (OTXTaskType.DETECTION, Label(label=0), False),  # Wrong type
            # INSTANCE_SEGMENTATION task tests
            (OTXTaskType.INSTANCE_SEGMENTATION, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.INSTANCE_SEGMENTATION, Bbox(x=0, y=0, w=-1, h=-1, label=0), False),  # Invalid bbox
            (OTXTaskType.INSTANCE_SEGMENTATION, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), True),
            (OTXTaskType.INSTANCE_SEGMENTATION, Polygon(points=[0, 0, 0, 0, 0, 0], label=0), False),  # Invalid polygon
            (OTXTaskType.INSTANCE_SEGMENTATION, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), True),
            (OTXTaskType.INSTANCE_SEGMENTATION, Label(label=0), False),  # Wrong type
            # Other task types (should use default is_valid_annot behavior)
            (OTXTaskType.MULTI_LABEL_CLS, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.MULTI_LABEL_CLS, Bbox(x=0, y=0, w=-1, h=-1, label=0), False),  # Invalid bbox
            (OTXTaskType.MULTI_LABEL_CLS, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), True),
            (OTXTaskType.MULTI_LABEL_CLS, Polygon(points=[0, 0, 0, 0, 0, 0], label=0), False),  # Invalid polygon
            (OTXTaskType.MULTI_LABEL_CLS, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), True),
            (OTXTaskType.MULTI_LABEL_CLS, Label(label=0), True),  # Label is always valid
            (OTXTaskType.SEMANTIC_SEGMENTATION, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.SEMANTIC_SEGMENTATION, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), True),
            (OTXTaskType.SEMANTIC_SEGMENTATION, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), True),
            (OTXTaskType.SEMANTIC_SEGMENTATION, Label(label=0), True),
            (OTXTaskType.ANOMALY, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.ANOMALY, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), True),
            (OTXTaskType.ANOMALY, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), True),
            (OTXTaskType.ROTATED_DETECTION, Bbox(x=0, y=0, w=10, h=10, label=0), True),
            (OTXTaskType.ROTATED_DETECTION, Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0), True),
            (OTXTaskType.ROTATED_DETECTION, Ellipse(x1=0, y1=0, x2=10, y2=10, label=0), True),
            (OTXTaskType.ROTATED_DETECTION, Label(label=0), False),
            # KEYPOINT_DETECTION task tests
            (
                OTXTaskType.KEYPOINT_DETECTION,
                Points(points=[10, 20, 30, 40], label=0),
                True,
            ),  # 2 keypoints, will use 2 labels
            (
                OTXTaskType.KEYPOINT_DETECTION,
                Points(points=[10, 20, 30, 40, 50, 60], label=0),
                True,
            ),  # 3 keypoints, will use 3 labels
            (OTXTaskType.KEYPOINT_DETECTION, Points(points=[10, 20], label=0), True),  # 1 keypoint, will use 1 label
            (OTXTaskType.KEYPOINT_DETECTION, Points(points=[], label=0), False),  # 0 keypoints, will use 0 labels
            (OTXTaskType.KEYPOINT_DETECTION, Bbox(x=0, y=0, w=10, h=10, label=0), False),  # Wrong type
            (
                OTXTaskType.KEYPOINT_DETECTION,
                Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0),
                False,
            ),  # Wrong type
            (OTXTaskType.KEYPOINT_DETECTION, Label(label=0), False),  # Wrong type
        ],
    )
    def test_is_valid_anno_for_task(
        self,
        fxt_dataset_item: DatasetItem,
        task: OTXTaskType,
        annotation,
        expected: bool,
    ) -> None:
        """Test is_valid_anno_for_task with various task types and annotations.

        Args:
            fxt_dataset_item: The dataset item to test with
            task: The task type to test
            annotation: The annotation to test
            expected: Expected result (True if valid, False if invalid)
        """
        # For keypoint detection, we need to provide the correct number of labels
        # based on the number of keypoints in the annotation
        if task == OTXTaskType.KEYPOINT_DETECTION and isinstance(annotation, Points):
            # Calculate expected number of labels based on points (each keypoint is x,y pair)
            expected_labels = len(annotation.points) // 2
            labels = [f"keypoint_{i}" for i in range(expected_labels)]
        else:
            labels = [0]

        result = is_valid_anno_for_task(fxt_dataset_item, annotation, task, labels)
        assert result == expected, f"Expected {expected} for task {task} with annotation {type(annotation).__name__}"

    def test_detection_task_with_valid_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with valid bounding box."""
        bbox = Bbox(x=5, y=5, w=20, h=15, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION, [0])
        assert result is True

    def test_detection_task_with_invalid_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with invalid bounding box (negative dimensions)."""
        bbox = Bbox(x=10, y=10, w=-5, h=-5, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION, [0])
        assert result is False

    def test_detection_task_with_zero_dimension_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with zero dimension bounding box."""
        bbox = Bbox(x=10, y=10, w=0, h=0, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION, [0])
        assert result is False

    def test_detection_task_with_wrong_annotation_type(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with non-bbox annotation types."""
        polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        ellipse = Ellipse(x1=0, y1=0, x2=10, y2=10, label=0)
        label = Label(label=0)

        assert is_valid_anno_for_task(fxt_dataset_item, polygon, OTXTaskType.DETECTION, [0]) is False
        assert is_valid_anno_for_task(fxt_dataset_item, ellipse, OTXTaskType.DETECTION, [0]) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.DETECTION, [0]) is False

    def test_instance_segmentation_task_with_valid_annotations(self, fxt_dataset_item: DatasetItem) -> None:
        """Test INSTANCE_SEGMENTATION task with valid annotation types."""
        bbox = Bbox(x=0, y=0, w=10, h=10, label=0)
        polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        ellipse = Ellipse(x1=0, y1=0, x2=10, y2=10, label=0)

        assert is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is True
        assert is_valid_anno_for_task(fxt_dataset_item, polygon, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is True
        assert is_valid_anno_for_task(fxt_dataset_item, ellipse, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is True

    def test_instance_segmentation_task_with_invalid_annotations(self, fxt_dataset_item: DatasetItem) -> None:
        """Test INSTANCE_SEGMENTATION task with invalid annotation types."""
        invalid_bbox = Bbox(x=0, y=0, w=-1, h=-1, label=0)
        invalid_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)  # Degenerate polygon
        label = Label(label=0)  # Wrong type

        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is False
        assert (
            is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is False
        )
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is False

    def test_other_task_types_use_default_validation(self, fxt_dataset_item: DatasetItem) -> None:
        """Test that other task types use the default is_valid_annot behavior."""
        valid_bbox = Bbox(x=0, y=0, w=10, h=10, label=0)
        invalid_bbox = Bbox(x=0, y=0, w=-1, h=-1, label=0)
        valid_polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        invalid_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)
        label = Label(label=0)

        # Test with CLASSIFICATION task
        assert is_valid_anno_for_task(fxt_dataset_item, valid_bbox, OTXTaskType.MULTI_CLASS_CLS, [0]) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.MULTI_CLASS_CLS, [0]) is False
        assert is_valid_anno_for_task(fxt_dataset_item, valid_polygon, OTXTaskType.MULTI_CLASS_CLS, [0]) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.MULTI_CLASS_CLS, [0]) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.MULTI_CLASS_CLS, [0]) is True

        # Test with SEMANTIC_SEGMENTATION task
        assert is_valid_anno_for_task(fxt_dataset_item, valid_bbox, OTXTaskType.SEMANTIC_SEGMENTATION, [0]) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.SEMANTIC_SEGMENTATION, [0]) is False
        assert is_valid_anno_for_task(fxt_dataset_item, valid_polygon, OTXTaskType.SEMANTIC_SEGMENTATION, [0]) is True
        assert (
            is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.SEMANTIC_SEGMENTATION, [0]) is False
        )
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.SEMANTIC_SEGMENTATION, [0]) is True

    def test_edge_cases(self, fxt_dataset_item: DatasetItem) -> None:
        """Test edge cases for annotation validation."""
        # Very small but valid bbox
        small_bbox = Bbox(x=0, y=0, w=0.1, h=0.1, label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, small_bbox, OTXTaskType.DETECTION, [0]) is True

        # Bbox with equal coordinates (should be invalid)
        equal_bbox = Bbox(x=5, y=5, w=0, h=0, label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, equal_bbox, OTXTaskType.DETECTION, [0]) is False

        # Polygon with minimal valid area
        minimal_polygon = Polygon(points=[0, 0, 1, 0, 1, 1, 0, 1], label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, minimal_polygon, OTXTaskType.INSTANCE_SEGMENTATION, [0]) is True

        # Degenerate polygon (should be invalid)
        degenerate_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)
        assert (
            is_valid_anno_for_task(fxt_dataset_item, degenerate_polygon, OTXTaskType.INSTANCE_SEGMENTATION, [0])
            is False
        )

    def test_keypoint_detection_task_with_valid_points(self, fxt_dataset_item: DatasetItem) -> None:
        """Test KEYPOINT_DETECTION task with valid Points annotations."""
        # Test with 2 keypoints (4 coordinates: x1, y1, x2, y2)
        points_2_kp = Points(points=[10, 20, 30, 40], label=0)
        labels_2 = ["left_eye", "right_eye"]
        result = is_valid_anno_for_task(fxt_dataset_item, points_2_kp, OTXTaskType.KEYPOINT_DETECTION, labels_2)
        assert result is True

        # Test with 4 keypoints (8 coordinates: x1, y1, x2, y2, x3, y3, x4, y4)
        points_4_kp = Points(points=[10, 20, 30, 40, 50, 60, 70, 80], label=0)
        labels_4 = ["left_eye", "right_eye", "nose", "mouth"]
        result = is_valid_anno_for_task(fxt_dataset_item, points_4_kp, OTXTaskType.KEYPOINT_DETECTION, labels_4)
        assert result is True

        # Test with single keypoint (2 coordinates: x1, y1)
        points_1_kp = Points(points=[10, 20], label=0)
        labels_1 = ["center"]
        result = is_valid_anno_for_task(fxt_dataset_item, points_1_kp, OTXTaskType.KEYPOINT_DETECTION, labels_1)
        assert result is True

    def test_keypoint_detection_task_with_invalid_points(self, fxt_dataset_item: DatasetItem) -> None:
        """Test KEYPOINT_DETECTION task with invalid Points annotations."""
        # Test with empty points
        empty_points = Points(points=[], label=0)
        labels = ["keypoint1", "keypoint2"]
        result = is_valid_anno_for_task(fxt_dataset_item, empty_points, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

        # Test with wrong number of keypoints (too many)
        too_many_points = Points(points=[10, 20, 30, 40, 50, 60], label=0)  # 3 keypoints
        labels = ["keypoint1", "keypoint2"]  # Only 2 labels
        result = is_valid_anno_for_task(fxt_dataset_item, too_many_points, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

        # Test with wrong number of keypoints (too few)
        too_few_points = Points(points=[10, 20], label=0)  # 1 keypoint
        labels = ["keypoint1", "keypoint2", "keypoint3"]  # 3 labels
        result = is_valid_anno_for_task(fxt_dataset_item, too_few_points, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

    def test_keypoint_detection_task_with_wrong_annotation_types(self, fxt_dataset_item: DatasetItem) -> None:
        """Test KEYPOINT_DETECTION task with non-Points annotation types."""
        labels = ["keypoint1", "keypoint2"]

        # Test with bbox (should be invalid)
        bbox = Bbox(x=0, y=0, w=10, h=10, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

        # Test with polygon (should be invalid)
        polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, polygon, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

        # Test with ellipse (should be invalid)
        ellipse = Ellipse(x1=0, y1=0, x2=10, y2=10, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, ellipse, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

        # Test with label (should be invalid)
        label = Label(label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is False

    def test_keypoint_detection_edge_cases(self, fxt_dataset_item: DatasetItem) -> None:
        """Test edge cases for keypoint detection validation."""
        # Test with zero coordinates (empty points)
        empty_points = Points(points=[], label=0)
        empty_labels = []
        result = is_valid_anno_for_task(fxt_dataset_item, empty_points, OTXTaskType.KEYPOINT_DETECTION, empty_labels)
        assert result is False  # Empty points should be invalid

        # Test with many keypoints
        many_points = Points(points=list(range(34)), label=0)  # 17 keypoints (34 coordinates)
        many_labels = [f"keypoint_{i}" for i in range(17)]
        result = is_valid_anno_for_task(fxt_dataset_item, many_points, OTXTaskType.KEYPOINT_DETECTION, many_labels)
        assert result is True

        # Test with negative coordinates (should still be valid as coordinates can be negative)
        negative_points = Points(points=[-10, -20, -30, -40], label=0)
        labels = ["keypoint1", "keypoint2"]
        result = is_valid_anno_for_task(fxt_dataset_item, negative_points, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is True

        # Test with floating point coordinates
        float_points = Points(points=[10.5, 20.7, 30.1, 40.9], label=0)
        labels = ["keypoint1", "keypoint2"]
        result = is_valid_anno_for_task(fxt_dataset_item, float_points, OTXTaskType.KEYPOINT_DETECTION, labels)
        assert result is True
