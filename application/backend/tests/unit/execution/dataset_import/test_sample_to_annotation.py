# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import numpy as np
import pytest
from datumaro.experimental.categories import LabelCategories

from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.execution.dataset_import.sample_to_annotation import DatumaroSampleToGetiAnnotationConverter
from app.models import FullImage, Label, Polygon, Rectangle


@pytest.fixture
def fxt_project_labels() -> list[Label]:
    """Create sample project labels."""
    return [
        Label(id=uuid4(), name="cat", color="#FF0000", hotkey=None),
        Label(id=uuid4(), name="dog", color="#00FF00", hotkey=None),
        Label(id=uuid4(), name="bird", color="#0000FF", hotkey=None),
    ]


@pytest.fixture
def fxt_project_labels_for_mapping() -> list[Label]:
    """Create sample project labels for mapping scenario."""
    return [
        Label(id=uuid4(), name="feline", color="#FF0000", hotkey=None),
        Label(id=uuid4(), name="canine", color="#00FF00", hotkey=None),
    ]


@pytest.fixture
def fxt_label_categories() -> LabelCategories:
    """Create Datumaro label categories."""
    return LabelCategories(
        labels=(
            "cat",
            "dog",
            "bird",
        )
    )


@pytest.fixture
def fxt_label_mapping() -> dict[str, str | None]:
    """Create sample label mapping with filtered label."""
    return {"cat": "feline", "dog": "canine", "bird": None}


@pytest.fixture
def fxt_converter(fxt_project_labels, fxt_label_categories) -> DatumaroSampleToGetiAnnotationConverter:
    """Create converter instance without label mapping."""
    return DatumaroSampleToGetiAnnotationConverter(fxt_project_labels, fxt_label_categories, None)


@pytest.fixture
def fxt_converter_with_mapping(
    fxt_project_labels_for_mapping, fxt_label_categories, fxt_label_mapping
) -> DatumaroSampleToGetiAnnotationConverter:
    """Create converter instance with label mapping."""
    return DatumaroSampleToGetiAnnotationConverter(
        fxt_project_labels_for_mapping, fxt_label_categories, fxt_label_mapping
    )


def test_convert_with_none(fxt_converter, fxt_project_labels):
    """Test converting a multilabel sample with confidence scores."""
    none_array = np.array(None)
    samples = [
        MultilabelClassificationImportExportSample(label=none_array, confidence=None),
        DetectionImportExportSample(label=none_array, bboxes=none_array, confidence=None),
        InstanceSegmentationImportExportSample(label=none_array, polygons=none_array, confidence=None),
    ]
    for sample in samples:
        with pytest.raises(ValueError, match="Expected 1D array for label indices, got 0D"):
            fxt_converter.convert_sample(sample)


class TestClassificationConversion:
    def test_convert_classification_sample_with_confidence(self, fxt_converter, fxt_project_labels):
        """Test converting a classification sample with confidence score."""
        sample = MulticlassClassificationImportExportSample(label=0, confidence=0.95)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert isinstance(result[0].shape, FullImage)
        assert len(result[0].labels) == 1
        assert result[0].labels[0].id == fxt_project_labels[0].id
        assert result[0].confidences == [0.95]

    def test_convert_classification_sample_without_confidence(self, fxt_converter, fxt_project_labels):
        """Test converting a classification sample without confidence."""
        sample = MulticlassClassificationImportExportSample(label=1, confidence=None)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert result[0].labels[0].id == fxt_project_labels[1].id
        assert result[0].confidences is None

    def test_convert_classification_sample_with_none_label(self, fxt_converter):
        """Test converting a classification sample with None label."""
        sample = MulticlassClassificationImportExportSample(label=None, confidence=0.5)

        result = fxt_converter.convert_sample(sample)

        assert result is None

    def test_convert_classification_sample_with_label_mapping(
        self, fxt_project_labels_for_mapping, fxt_converter_with_mapping
    ):
        """Test converting a classification sample with label mapping."""
        sample = MulticlassClassificationImportExportSample(label=0, confidence=0.9)
        result = fxt_converter_with_mapping.convert_sample(sample)

        assert result is not None
        assert len(result) == 1
        assert result[0].labels[0].id == fxt_project_labels_for_mapping[0].id

    def test_convert_with_label_filtering_success(self, fxt_project_labels, fxt_label_categories):
        """Test label name mapped to None during conversion."""
        label_mapping: dict[str, str | None] = {"bird": None}
        project_labels = fxt_project_labels[:2]  # Only "cat" and "dog" labels in project
        converter = DatumaroSampleToGetiAnnotationConverter(project_labels, fxt_label_categories, label_mapping)

        sample = MulticlassClassificationImportExportSample(label=2, confidence=0.9)
        result = converter.convert_sample(sample)

        assert result is None  # Label should be filtered out and no annotations returned


class TestMultilabelConversion:
    def test_convert_multilabel_sample_with_confidences(self, fxt_converter, fxt_project_labels):
        """Test converting a multilabel sample with confidence scores."""
        labels = np.array([0, 1, 2])
        confidences = np.array([0.9, 0.85, 0.8])
        sample = MultilabelClassificationImportExportSample(label=labels, confidence=confidences)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert isinstance(result[0].shape, FullImage)
        assert len(result[0].labels) == 3
        assert result[0].labels[0].id == fxt_project_labels[0].id
        assert result[0].labels[1].id == fxt_project_labels[1].id
        assert result[0].labels[2].id == fxt_project_labels[2].id
        np.testing.assert_array_equal(result[0].confidences, confidences)

    def test_convert_multilabel_sample_without_confidences(self, fxt_converter):
        """Test converting a multilabel sample without confidences."""
        labels = np.array([1])
        sample = MultilabelClassificationImportExportSample(label=labels, confidence=None)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert len(result[0].labels) == 1
        assert result[0].confidences is None

    def test_convert_multilabel_sample_with_label_mapping(
        self, fxt_project_labels_for_mapping, fxt_converter_with_mapping
    ):
        """Test converting a multilabel sample with label mapping."""
        sample = MultilabelClassificationImportExportSample(label=np.array([0, 1]), confidence=np.array([0.9] * 2))
        result = fxt_converter_with_mapping.convert_sample(sample)

        assert result is not None
        assert len(result) == 1
        assert result[0].labels[0].id == fxt_project_labels_for_mapping[0].id


class TestDetectionConversion:
    def test_convert_detection_sample(self, fxt_converter, fxt_project_labels):
        """Test converting a detection sample with bounding boxes."""
        labels = np.array([0, 1])
        bboxes = np.array([[10, 20, 50, 60], [100, 150, 200, 250]])
        confidences = np.array([0.95, 0.88])
        sample = DetectionImportExportSample(label=labels, bboxes=bboxes, confidence=confidences)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 2

        assert isinstance(result[0].shape, Rectangle)
        assert result[0].shape.x == 10
        assert result[0].shape.y == 20
        assert result[0].shape.width == 40
        assert result[0].shape.height == 40
        assert result[0].labels[0].id == fxt_project_labels[0].id
        assert result[0].confidences == pytest.approx([0.95], abs=1e-6)

        assert result[1].shape.x == 100
        assert result[1].labels[0].id == fxt_project_labels[1].id

    def test_convert_detection_sample_without_confidences(self, fxt_converter):
        """Test converting a detection sample without confidences."""
        labels = np.array([2])
        bboxes = np.array([[5, 10, 15, 20]])
        sample = DetectionImportExportSample(label=labels, bboxes=bboxes, confidence=None)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert result[0].confidences is None

    def test_convert_detection_with_label_mapping_success(
        self, fxt_project_labels_for_mapping, fxt_converter_with_mapping
    ):
        """Test converting a detection sample with label mapping and multiple bounding boxes."""
        labels = np.array([1, 2, 0])
        bboxes = np.array([[10, 20, 50, 60]] * 3)
        confidences = np.array([0.92] * 3)
        sample = DetectionImportExportSample(label=labels, bboxes=bboxes, confidence=confidences)
        result = fxt_converter_with_mapping.convert_sample(sample)

        assert result is not None
        assert len(result) == 2
        assert result[0].labels[0].id == fxt_project_labels_for_mapping[1].id
        assert result[1].labels[0].id == fxt_project_labels_for_mapping[0].id


class TestSegmentationConversion:
    def test_convert_segmentation_sample(self, fxt_converter, fxt_project_labels):
        """Test converting an instance segmentation sample."""
        labels = np.array([0])
        polygons = np.array([[[10.2, 20.5], [30.6, 40.7], [50.1, 60.3]]])
        confidences = np.array([0.92])
        sample = InstanceSegmentationImportExportSample(label=labels, polygons=polygons, confidence=confidences)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 1
        assert isinstance(result[0].shape, Polygon)
        assert len(result[0].shape.points) == 3
        assert result[0].shape.points[0].x == 10.2
        assert result[0].shape.points[0].y == 20.5
        assert result[0].labels[0].id == fxt_project_labels[0].id
        assert result[0].confidences == pytest.approx([0.92], abs=1e-6)

    def test_convert_segmentation_sample_multiple_polygons(self, fxt_converter):
        """Test converting multiple instance segmentation polygons."""
        labels = np.array([1, 2])
        polygons = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], [[20.0, 20.0], [30.0, 30.0]]], dtype=object)
        sample = InstanceSegmentationImportExportSample(label=labels, polygons=polygons, confidence=None)

        result = fxt_converter.convert_sample(sample)

        assert len(result) == 2
        assert len(result[0].shape.points) == 3
        assert len(result[1].shape.points) == 2

    def test_convert_segmentation_with_label_mapping_success(
        self, fxt_project_labels_for_mapping, fxt_converter_with_mapping
    ):
        """Test converting a segmentation sample with label mapping and multiple polygons."""
        labels = np.array([1, 2, 0])
        polygons = np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]] * 3)
        confidences = np.array([0.92] * 3)
        sample = InstanceSegmentationImportExportSample(label=labels, polygons=polygons, confidence=confidences)
        result = fxt_converter_with_mapping.convert_sample(sample)

        assert result is not None
        assert len(result) == 2
        assert result[0].labels[0].id == fxt_project_labels_for_mapping[1].id
        assert result[1].labels[0].id == fxt_project_labels_for_mapping[0].id


class TestLabelMapping:
    def test_converter_with_labels_not_in_project(self, fxt_label_categories, fxt_project_labels):
        """Test raises error when instantiating converter with labels not found in project."""
        project_labels = [fxt_project_labels[0]]  # Only "cat" label in project

        with pytest.raises(ValueError, match="Unmapped labels not in project: \['dog', 'bird'\]"):
            DatumaroSampleToGetiAnnotationConverter(project_labels, fxt_label_categories, None)

    def test_converter_with_mappings_not_in_project(self, fxt_label_categories, fxt_project_labels):
        """Test raises error when instantiating converter with mapping not found in project."""
        project_labels = [fxt_project_labels[0]]  # Only "cat" label in project
        label_mapping = {"cat": "feline", "dog": None, "bird": None}

        with pytest.raises(ValueError, match="Mapped labels with invalid targets: \[\('cat', 'feline'\)\]"):
            DatumaroSampleToGetiAnnotationConverter(project_labels, fxt_label_categories, label_mapping)
