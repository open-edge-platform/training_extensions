# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo

from app.services.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)


@pytest.fixture(scope="module")
def fxt_multiclass_classification_dataset_gt() -> Dataset:
    # Ground truth dataset for multiclass classification task
    dataset = Dataset(ClassificationSample, categories={"label": LabelCategories(labels=("cat", "dog", "bird"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        ClassificationSample(image="/dummy/path/A.jpg", image_info=img_info, label=0, confidence=None),
        ClassificationSample(image="/dummy/path/B.jpg", image_info=img_info, label=1, confidence=None),
        ClassificationSample(image="/dummy/path/C.jpg", image_info=img_info, label=2, confidence=None),
        ClassificationSample(image="/dummy/path/D.jpg", image_info=img_info, label=1, confidence=None),
        ClassificationSample(image="/dummy/path/E.jpg", image_info=img_info, label=2, confidence=None),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_multiclass_classification_dataset_pred() -> Dataset:
    # Prediction dataset for multiclass classification task
    dataset = Dataset(ClassificationSample, categories={"label": LabelCategories(labels=("cat", "dog", "bird"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        ClassificationSample(image="/dummy/path/A.jpg", image_info=img_info, label=0, confidence=0.9),  # correct
        ClassificationSample(image="/dummy/path/B.jpg", image_info=img_info, label=2, confidence=0.6),  # wrong
        ClassificationSample(image="/dummy/path/C.jpg", image_info=img_info, label=1, confidence=0.5),  # wrong
        ClassificationSample(image="/dummy/path/D.jpg", image_info=img_info, label=1, confidence=0.8),  # correct
        ClassificationSample(image="/dummy/path/E.jpg", image_info=img_info, label=2, confidence=0.9),  # correct
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_multilabel_classification_dataset_gt() -> Dataset:
    # Ground truth dataset for multilabel classification task
    dataset = Dataset(
        MultilabelClassificationSample, categories={"label": LabelCategories(labels=("pop", "rock", "jazz"))}
    )
    img_info = ImageInfo(width=100, height=100)
    samples = (
        MultilabelClassificationSample(
            image="/dummy/path/A.jpg", image_info=img_info, label=np.array([0, 1]), confidence=None
        ),
        MultilabelClassificationSample(
            image="/dummy/path/B.jpg", image_info=img_info, label=np.array([1]), confidence=None
        ),
        MultilabelClassificationSample(
            image="/dummy/path/C.jpg", image_info=img_info, label=np.array([2, 0]), confidence=None
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_multilabel_classification_dataset_pred() -> Dataset:
    # Prediction dataset for multilabel classification task
    dataset = Dataset(
        MultilabelClassificationSample, categories={"label": LabelCategories(labels=("pop", "rock", "jazz"))}
    )
    img_info = ImageInfo(width=100, height=100)
    samples = (
        MultilabelClassificationSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            label=np.array([0]),
            confidence=np.array([0.85]),
        ),  # missing one label
        MultilabelClassificationSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            label=np.array([1, 2]),
            confidence=np.array([0.8, 0.6]),
        ),  # one extra label
        MultilabelClassificationSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            label=np.array([2, 0]),
            confidence=np.array([0.9, 0.7]),
        ),  # correct
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_detection_dataset_gt() -> Dataset:
    # Ground truth dataset for detection task
    dataset = Dataset(DetectionSample, categories={"label": LabelCategories(labels=("car", "person"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        DetectionSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            bboxes=np.array([[10, 15, 30, 35]]),
            label=np.array([1]),
            confidence=None,
        ),
        DetectionSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            bboxes=np.array([[5, 5, 20, 20], [25, 30, 50, 60]]),
            label=np.array([0, 1]),
            confidence=None,
        ),
        DetectionSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            bboxes=np.array([[0, 0, 15, 15]]),
            label=np.array([0]),
            confidence=None,
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_detection_dataset_pred() -> Dataset:
    # Prediction dataset for detection task
    dataset = Dataset(DetectionSample, categories={"label": LabelCategories(labels=("car", "person"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        DetectionSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            bboxes=np.array([[10, 20, 30, 40]]),  # partial overlap (IoU = 0.6)
            label=np.array([1]),  # correct
            confidence=np.array([0.8]),
        ),
        DetectionSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            bboxes=np.array([[5, 5, 20, 20], [25, 30, 50, 60]]),  # correct
            label=np.array([0, 1]),  # correct
            confidence=np.array([0.9, 0.7]),
        ),
        DetectionSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            bboxes=np.array([[0, 0, 15, 15]]),  # correct
            label=np.array([1]),  # wrong
            confidence=np.array([0.6]),
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_instance_segmentation_dataset_gt() -> Dataset:
    # Ground truth dataset for instance segmentation task
    dataset = Dataset(InstanceSegmentationSample, categories={"label": LabelCategories(labels=("apple", "banana"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        InstanceSegmentationSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            polygons=np.array([[[10, 20], [30, 40], [40, 70], [10, 60]], [[10, 20], [30, 40], [50, 40]]], dtype=object),
            label=np.array([0, 1]),
            confidence=None,
        ),
        InstanceSegmentationSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            polygons=np.array([[[50, 50], [90, 50], [50, 80]]]),
            label=np.array([0]),
            confidence=None,
        ),
        InstanceSegmentationSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            polygons=np.array([[[15, 15], [25, 15], [25, 25], [15, 25]]]),
            label=np.array([1]),
            confidence=None,
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture(scope="module")
def fxt_instance_segmentation_dataset_pred() -> Dataset:
    # Prediction dataset for instance segmentation task
    dataset = Dataset(InstanceSegmentationSample, categories={"label": LabelCategories(labels=("apple", "banana"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        InstanceSegmentationSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            polygons=np.array(
                [[[10, 20], [30, 40], [40, 70], [10, 60]], [[10, 20], [30, 40], [50, 40]]], dtype=object
            ),  # correct
            label=np.array([0, 1]),  # correct
            confidence=np.array([0.9, 0.75]),
        ),
        InstanceSegmentationSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            polygons=np.array([[[50, 50], [82, 50], [50, 74]]]),  # partial overlap (64% IoU)
            label=np.array([0]),  # correct
            confidence=np.array([0.8]),
        ),
        InstanceSegmentationSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            polygons=np.array([[[15, 15], [25, 15], [25, 25], [15, 25]]]),  # correct
            label=np.array([0]),  # wrong
            confidence=np.array([0.6]),
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset
