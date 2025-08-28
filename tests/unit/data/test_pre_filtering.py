# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.components.annotation import AnnotationType, Bbox, Ellipse, Label, Polygon
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
        result = is_valid_anno_for_task(fxt_dataset_item, annotation, task)
        assert result == expected, f"Expected {expected} for task {task} with annotation {type(annotation).__name__}"

    def test_detection_task_with_valid_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with valid bounding box."""
        bbox = Bbox(x=5, y=5, w=20, h=15, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION)
        assert result is True

    def test_detection_task_with_invalid_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with invalid bounding box (negative dimensions)."""
        bbox = Bbox(x=10, y=10, w=-5, h=-5, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION)
        assert result is False

    def test_detection_task_with_zero_dimension_bbox(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with zero dimension bounding box."""
        bbox = Bbox(x=10, y=10, w=0, h=0, label=0)
        result = is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.DETECTION)
        assert result is False

    def test_detection_task_with_wrong_annotation_type(self, fxt_dataset_item: DatasetItem) -> None:
        """Test DETECTION task with non-bbox annotation types."""
        polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        ellipse = Ellipse(x1=0, y1=0, x2=10, y2=10, label=0)
        label = Label(label=0)

        assert is_valid_anno_for_task(fxt_dataset_item, polygon, OTXTaskType.DETECTION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, ellipse, OTXTaskType.DETECTION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.DETECTION) is False

    def test_instance_segmentation_task_with_valid_annotations(self, fxt_dataset_item: DatasetItem) -> None:
        """Test INSTANCE_SEGMENTATION task with valid annotation types."""
        bbox = Bbox(x=0, y=0, w=10, h=10, label=0)
        polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        ellipse = Ellipse(x1=0, y1=0, x2=10, y2=10, label=0)

        assert is_valid_anno_for_task(fxt_dataset_item, bbox, OTXTaskType.INSTANCE_SEGMENTATION) is True
        assert is_valid_anno_for_task(fxt_dataset_item, polygon, OTXTaskType.INSTANCE_SEGMENTATION) is True
        assert is_valid_anno_for_task(fxt_dataset_item, ellipse, OTXTaskType.INSTANCE_SEGMENTATION) is True

    def test_instance_segmentation_task_with_invalid_annotations(self, fxt_dataset_item: DatasetItem) -> None:
        """Test INSTANCE_SEGMENTATION task with invalid annotation types."""
        invalid_bbox = Bbox(x=0, y=0, w=-1, h=-1, label=0)
        invalid_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)  # Degenerate polygon
        label = Label(label=0)  # Wrong type

        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.INSTANCE_SEGMENTATION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.INSTANCE_SEGMENTATION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.INSTANCE_SEGMENTATION) is False

    def test_other_task_types_use_default_validation(self, fxt_dataset_item: DatasetItem) -> None:
        """Test that other task types use the default is_valid_annot behavior."""
        valid_bbox = Bbox(x=0, y=0, w=10, h=10, label=0)
        invalid_bbox = Bbox(x=0, y=0, w=-1, h=-1, label=0)
        valid_polygon = Polygon(points=[0, 0, 10, 0, 10, 10, 0, 10], label=0)
        invalid_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)
        label = Label(label=0)

        # Test with CLASSIFICATION task
        assert is_valid_anno_for_task(fxt_dataset_item, valid_bbox, OTXTaskType.MULTI_CLASS_CLS) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.MULTI_CLASS_CLS) is False
        assert is_valid_anno_for_task(fxt_dataset_item, valid_polygon, OTXTaskType.MULTI_CLASS_CLS) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.MULTI_CLASS_CLS) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.MULTI_CLASS_CLS) is True

        # Test with SEMANTIC_SEGMENTATION task
        assert is_valid_anno_for_task(fxt_dataset_item, valid_bbox, OTXTaskType.SEMANTIC_SEGMENTATION) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_bbox, OTXTaskType.SEMANTIC_SEGMENTATION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, valid_polygon, OTXTaskType.SEMANTIC_SEGMENTATION) is True
        assert is_valid_anno_for_task(fxt_dataset_item, invalid_polygon, OTXTaskType.SEMANTIC_SEGMENTATION) is False
        assert is_valid_anno_for_task(fxt_dataset_item, label, OTXTaskType.SEMANTIC_SEGMENTATION) is True

    def test_edge_cases(self, fxt_dataset_item: DatasetItem) -> None:
        """Test edge cases for annotation validation."""
        # Very small but valid bbox
        small_bbox = Bbox(x=0, y=0, w=0.1, h=0.1, label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, small_bbox, OTXTaskType.DETECTION) is True

        # Bbox with equal coordinates (should be invalid)
        equal_bbox = Bbox(x=5, y=5, w=0, h=0, label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, equal_bbox, OTXTaskType.DETECTION) is False

        # Polygon with minimal valid area
        minimal_polygon = Polygon(points=[0, 0, 1, 0, 1, 1, 0, 1], label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, minimal_polygon, OTXTaskType.INSTANCE_SEGMENTATION) is True

        # Degenerate polygon (should be invalid)
        degenerate_polygon = Polygon(points=[0, 0, 0, 0, 0, 0], label=0)
        assert is_valid_anno_for_task(fxt_dataset_item, degenerate_polygon, OTXTaskType.INSTANCE_SEGMENTATION) is False
