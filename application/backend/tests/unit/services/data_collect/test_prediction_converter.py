# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import uuid4

import cv2
import model_api.models.result
import numpy as np
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult

from app.schemas import Label
from app.schemas.dataset_item import DatasetItemAnnotation
from app.schemas.label import LabelReference
from app.schemas.shape import FullImage, Point, Polygon, Rectangle
from app.services.data_collect.prediction_converter import convert_prediction


def test_convert_prediction_classification() -> None:
    # Arrange
    frame_data = np.random.rand(100, 100, 3)
    label = Label(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
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
    label = Label(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
    coords = [12, 41, 12, 46]
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
            shape=Rectangle(x=12, y=41, width=12, height=46),
            confidence=0.81,
        )
    ]


def test_convert_prediction_segmentation() -> None:
    # Arrange
    frame_data = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (100, 100), 255, -1)

    label = Label(id=uuid4(), name="cat", color="#ff0000", hotkey="c")
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
