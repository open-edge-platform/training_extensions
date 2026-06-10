# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import pytest
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult
from model_api.models.result import Label
from model_api.visualizer import BoundingBox, Flatten, Polygon
from model_api.visualizer import Label as VisualizerLabel

from app.utils.visualization import (
    ClassificationVisualizerCreator,
    DetectionVisualizerCreator,
    InstanceSegmentationVisualizerCreator,
    VisualizationDispatcher,
    _compute_scale,
)

bboxes = np.array([[10, 20, 50, 60], [30, 40, 70, 80], [15, 25, 55, 65]], dtype=np.int32)

labels = np.array([1, 2, 3], dtype=np.int32)

masks = np.array(
    [
        np.zeros((100, 100), dtype=np.uint8),
        np.ones((100, 100), dtype=np.uint8),
        np.full((100, 100), 255, dtype=np.uint8),
    ],
    dtype=np.uint8,
)


class TestVisualizationDispatcherValidation(unittest.TestCase):
    def test_handles_empty_image_input(self):
        dispatcher = VisualizationDispatcher()
        original_image = np.array([])
        predictions = DetectionResult(bboxes, labels)
        with self.assertRaises(Exception):
            dispatcher.create_visualization(original_image, predictions)


class TestDetectionVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = DetectionVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))

    def test_uses_visualizer_label_in_layout(self):
        creator = DetectionVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        with mock.patch("app.utils.visualization.DetectionScene") as mock_scene:
            creator.create_visualization(original_image, predictions)
        layout = mock_scene.call_args.kwargs["layout"]
        self.assertIsInstance(layout, Flatten)
        self.assertEqual(layout.children, (BoundingBox, VisualizerLabel))


class TestInstanceSegmentationVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))

    def test_uses_visualizer_label_in_layout(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        with mock.patch("app.utils.visualization.InstanceSegmentationScene") as mock_scene:
            creator.create_visualization(original_image, predictions)
        layout = mock_scene.call_args.kwargs["layout"]
        self.assertIsInstance(layout, Flatten)
        self.assertEqual(layout.children, (Polygon, VisualizerLabel))


class TestClassificationVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = ClassificationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        classification_labels = [Label(id=1, name="1", confidence=0.9), Label(id=2, name="2", confidence=0.1)]
        predictions = ClassificationResult(classification_labels)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))


class TestVisualizationHelpers(unittest.TestCase):
    def test_compute_scale_handles_none_or_empty(self):
        assert _compute_scale(None) == 1.0  # type: ignore[arg-type]
        assert _compute_scale(np.array([])) == 1.0

    def test_compute_scale_never_below_one(self):
        small = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _compute_scale(small) == 1.0

    def test_compute_scale_scales_with_longer_edge(self):
        # 4K longer edge → ~3.0
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        assert _compute_scale(img) == pytest.approx(3.0, rel=1e-3)
