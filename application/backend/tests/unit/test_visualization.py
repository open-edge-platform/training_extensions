# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult
from model_api.models.result import Label

from app.utils.visualization import (
    ClassificationVisualizerCreator,
    DetectionVisualizerCreator,
    InstanceSegmentationVisualizerCreator,
    VisualizationDispatcher,
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


class TestInstanceSegmentationVisualizerCreator(unittest.TestCase):
    @pytest.mark.skip(reason="Disabled due to model api bug https://github.com/open-edge-platform/model_api/issues/328")
    def test_creates_visualization(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))


class TestClassificationVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = ClassificationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        classification_labels = [Label(id=1, name="1", confidence=0.9), Label(id=2, name="2", confidence=0.1)]
        predictions = ClassificationResult(classification_labels)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))
