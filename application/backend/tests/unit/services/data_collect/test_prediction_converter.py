# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import uuid4

import cv2
import model_api.models.result
import numpy as np
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult

from app.models import DatasetItemAnnotation, FullImage, Label, LabelReference, Point, Polygon, Rectangle
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
    project_id, label_1_id, label_2_id = uuid4(), uuid4(), uuid4()
    labels = [
        Label(id=label_1_id, project_id=project_id, name="dog", color="#00ff00", hotkey="d"),
        Label(id=label_2_id, project_id=project_id, name="cat", color="#ff0000", hotkey="c"),
    ]
    raw_prediction = ClassificationResult(  # multilabel sample with two labels
        top_labels=[
            model_api.models.result.Label(id=0, name=labels[0].name, confidence=0.81),
            model_api.models.result.Label(id=1, name=labels[1].name, confidence=0.65),
        ],
    )

    # Act
    annotations = convert_prediction(labels=labels, frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            shape=FullImage(),
            labels=[LabelReference(id=labels[0].id), LabelReference(id=labels[1].id)],
            confidences=[0.81, 0.65],
        )
    ]


def test_convert_prediction_detection() -> None:
    # Arrange
    frame_data = np.random.rand(100, 100, 3)
    project_id, label_1_id, label_2_id = uuid4(), uuid4(), uuid4()
    labels = [
        Label(id=label_1_id, project_id=project_id, name="cat", color="#ff0000", hotkey="c"),
        Label(id=label_2_id, project_id=project_id, name="dog", color="#00ff00", hotkey="d"),
    ]
    coords = [[12, 41, 30, 65], [130, 213, 164, 244]]  # bboxes in xyxy format
    raw_prediction = DetectionResult(
        bboxes=np.array(coords),
        labels=np.array([1, 0]),
        scores=np.array([0.81, 0.84]),
        label_names=["dog", "cat"],
    )

    # Act
    annotations = convert_prediction(labels=labels, frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            shape=Rectangle(x=12, y=41, width=18, height=24),
            labels=[LabelReference(id=labels[1].id)],
            confidences=[0.81],
        ),
        DatasetItemAnnotation(
            shape=Rectangle(x=130, y=213, width=34, height=31),
            labels=[LabelReference(id=labels[0].id)],
            confidences=[0.84],
        ),
    ]


def test_convert_prediction_segmentation() -> None:
    # Arrange
    frame_data = np.zeros((100, 200, 3), dtype=np.uint8)

    # First mask - rectangle in the left portion
    mask1 = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask1, (10, 20), (50, 70), 255, -1)

    # Second mask - rectangle in the right portion
    mask2 = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask2, (120, 30), (180, 80), 255, -1)

    label_cat = Label(id=uuid4(), project_id=uuid4(), name="cat", color="#ff0000", hotkey="c")
    label_dog = Label(id=uuid4(), project_id=uuid4(), name="dog", color="#00ff00", hotkey="d")

    # Bounding boxes corresponding to the masks (xyxy format)
    coords = [[10, 20, 50, 70], [120, 30, 180, 80]]
    raw_prediction = InstanceSegmentationResult(
        bboxes=np.array(coords),
        labels=np.array([0, 1]),
        masks=np.array([mask1, mask2]),
        scores=np.array([0.81, 0.92]),
        label_names=["cat", "dog"],
    )

    # Act
    annotations = convert_prediction(labels=[label_cat, label_dog], frame_data=frame_data, prediction=raw_prediction)

    # Assert
    assert annotations == [
        DatasetItemAnnotation(
            shape=Polygon(points=[Point(x=10, y=20), Point(x=10, y=70), Point(x=50, y=70), Point(x=50, y=20)]),
            labels=[LabelReference(id=label_cat.id)],
            confidences=[0.81],
        ),
        DatasetItemAnnotation(
            shape=Polygon(points=[Point(x=120, y=30), Point(x=120, y=80), Point(x=180, y=80), Point(x=180, y=30)]),
            labels=[LabelReference(id=label_dog.id)],
            confidences=[0.92],
        ),
    ]
