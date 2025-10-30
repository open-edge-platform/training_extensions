# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import uuid4

import cv2
import model_api.models.result
import numpy as np
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult

from app.models import DatasetItemAnnotation, FullImage, LabelReference, Point, Polygon, Rectangle
from app.schemas import LabelView
from app.services.data_collect.prediction_converter import convert_prediction, get_confidence_scores


def test_get_confidence_scores_classification() -> None:
    # Arrange
    raw_prediction = ClassificationResult(
        top_labels=[
            model_api.models.result.Label(id=1, name="cat", confidence=0.81),
            model_api.models.result.Label(id=2, name="dog", confidence=0.63),
        ],
        raw_scores=[0.19, 0.81],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    confidence_scores = get_confidence_scores(prediction=raw_prediction)

    # Assert
    assert confidence_scores == [0.81, 0.63]


def test_get_confidence_scores_detection() -> None:
    # Arrange
    raw_prediction = DetectionResult(
        bboxes=np.array([]),
        labels=np.array([1, 2]),
        scores=np.array([0.86, 0.62]),
        label_names=["cat", "dog"],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    confidence_scores = get_confidence_scores(prediction=raw_prediction)

    # Assert
    assert confidence_scores == [0.86, 0.62]


def test_get_confidence_scores_segmentation() -> None:
    # Arrange

    raw_prediction = InstanceSegmentationResult(
        bboxes=np.array([]),
        labels=np.array([1, 2]),
        masks=np.array([]),
        scores=np.array([0.32, 0.58]),
        label_names=["cat", "dog"],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    confidence_scores = get_confidence_scores(prediction=raw_prediction)

    # Assert
    assert confidence_scores == [0.32, 0.58]


def test_convert_prediction_classification() -> None:
    # Arrange
    frame_data = np.random.rand(100, 100, 3)
    label = LabelView(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
    raw_prediction = ClassificationResult(
        top_labels=[model_api.models.result.Label(id=1, name=label.name, confidence=0.81)],
        raw_scores=[0.19, 0.81],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    annotations = convert_prediction(labels=[label], frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            labels=[LabelReference(id=label.id)],
            shape=FullImage(),
            confidence=0.81,
        )
    ]


def test_convert_prediction_detection() -> None:
    # Arrange
    frame_data = np.random.rand(100, 100, 3)
    label = LabelView(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
    coords = [12, 41, 30, 65]  # x1, y1, x2, y2
    raw_prediction = DetectionResult(
        bboxes=np.array([coords]),
        labels=np.array([1]),
        scores=np.array([0.81]),
        label_names=["cat"],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    annotations = convert_prediction(labels=[label], frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            labels=[LabelReference(id=label.id)],
            shape=Rectangle(x=12, y=41, width=18, height=24),
            confidence=0.81,
        )
    ]


def test_convert_prediction_segmentation() -> None:
    # Arrange
    frame_data = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (100, 100), 255, -1)

    label = LabelView(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
    coords = [12, 41, 12, 46]
    raw_prediction = InstanceSegmentationResult(
        bboxes=np.array([coords]),
        labels=np.array([1]),
        masks=np.array([mask]),
        scores=np.array([0.81]),
        label_names=["cat"],
        saliency_map=None,
        feature_vector=None,
    )

    # Act
    annotations = convert_prediction(labels=[label], frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            labels=[LabelReference(id=label.id)],
            shape=Polygon(points=[Point(x=0, y=0), Point(x=0, y=99), Point(x=100, y=99), Point(x=100, y=0)]),
            confidence=0.81,
        )
    ]
