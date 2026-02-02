# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from loguru import logger
from model_api.models import ClassificationResult
from model_api.models.result import DetectionResult, InstanceSegmentationResult, Label, Result
from model_api.visualizer import BoundingBox, Flatten, Polygon
from model_api.visualizer.scene import ClassificationScene, DetectionScene, InstanceSegmentationScene
from PIL import Image

from app.utils.singleton import Singleton

R = TypeVar("R", bound=Result)


class VisualizerCreator(ABC, Generic[R]):
    """Abstract base class for visualizer creators."""

    @abstractmethod
    def create_visualization(self, original_image: np.ndarray, predictions: R) -> np.ndarray:
        """Create a visualization of the predictions on the original image."""


class ClassificationVisualizerCreator(VisualizerCreator[ClassificationResult]):
    """Creator for classification visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: ClassificationResult) -> np.ndarray:
        """Create a visualization of the classification predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        classification_scene = ClassificationScene(
            image=image_pil,
            result=predictions,
        )
        rendered_classification_pil = classification_scene.render()
        return np.array(rendered_classification_pil)


class DetectionVisualizerCreator(VisualizerCreator[DetectionResult]):
    """Creator for detection visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: DetectionResult) -> np.ndarray:
        """Create a visualization of the detection predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        detection_scene = DetectionScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(BoundingBox, Label),  # pyrefly: ignore[bad-argument-type]
        )
        rendered_detections_pil = detection_scene.render()
        return np.array(rendered_detections_pil)


class InstanceSegmentationVisualizerCreator(VisualizerCreator[InstanceSegmentationResult]):
    """Creator for instance segmentation visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: InstanceSegmentationResult) -> np.ndarray:
        """Create a visualization of the instance segmentation predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        segmentation_scene = InstanceSegmentationScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(Polygon, Label),  # pyrefly: ignore[bad-argument-type]
        )
        rendered_segmentation_pil = segmentation_scene.render()
        return np.array(rendered_segmentation_pil)


class VisualizationDispatcher(metaclass=Singleton):
    """Dispatcher for creating visualizations."""

    def __init__(self) -> None:
        self._creator_map: dict[type[Result], VisualizerCreator] = {
            DetectionResult: DetectionVisualizerCreator(),
            ClassificationResult: ClassificationVisualizerCreator(),
            InstanceSegmentationResult: InstanceSegmentationVisualizerCreator(),
        }

    def create_visualization(self, original_image: np.ndarray, predictions: Result) -> np.ndarray | None:
        """Create a visualization of the predictions on the original image."""
        if original_image.size == 0:
            raise ValueError("The image provided through the 'original_image' parameter cannot be empty.")

        creator = self._creator_map.get(type(predictions))
        if creator is not None:
            return creator.create_visualization(original_image, predictions)
        logger.error("Visualization for {} is not supported.", type(predictions))
        return None


class Visualizer:
    @staticmethod
    def overlay_predictions(original_image: np.ndarray, predictions: Result) -> np.ndarray:
        """Overlay predictions on the original image based on the type of predictions."""
        try:
            visualization = VisualizationDispatcher().create_visualization(original_image, predictions)
            if visualization is None:
                # If no visualization could be created, return the original image
                return original_image
        except Exception:
            logger.exception("An error occurred while creating visualization, returning original image.")
            return original_image
        return visualization
