# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PIL import Image

from app.utils.singleton import Singleton

if TYPE_CHECKING:
    from model_api.models.result import ClassificationResult, DetectionResult, InstanceSegmentationResult, Result


def _compute_scale(image: np.ndarray) -> float:
    """Compute a font/outline scale factor based on the image's longer edge.

    Uses SCALE_BASELINE (720p longer edge = 1280) as the reference: at 1280px the
    scale is 1.0; at 4K (3840px) the scale is ~3.0. Never shrinks below 1.0.
    """
    from model_api.visualizer.defaults import SCALE_BASELINE

    if image is None or image.size == 0:
        return 1.0
    h, w = image.shape[:2]
    longer_edge = float(max(int(h), int(w)))
    return max(1.0, longer_edge / float(SCALE_BASELINE))


class VisualizerCreator(ABC):
    """Abstract base class for visualizer creators."""

    @abstractmethod
    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: "Result",
    ) -> np.ndarray:
        """Create a visualization of the predictions on the original image."""


class ClassificationVisualizerCreator(VisualizerCreator):
    """Creator for classification visualizations."""

    def create_visualization(  # pyrefly: ignore[bad-override]
        self,
        original_image: np.ndarray,
        predictions: "ClassificationResult",
    ) -> np.ndarray:
        from model_api.visualizer.scene import ClassificationScene

        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        classification_scene = ClassificationScene(
            image=image_pil,
            result=predictions,
            scale=scale,
        )
        rendered = classification_scene.render()
        return np.array(rendered)


class DetectionVisualizerCreator(VisualizerCreator):
    """Creator for detection visualizations."""

    def create_visualization(  # pyrefly: ignore[bad-override]
        self,
        original_image: np.ndarray,
        predictions: "DetectionResult",
    ) -> np.ndarray:
        from model_api.models.result import Label
        from model_api.visualizer import BoundingBox, Flatten
        from model_api.visualizer.scene import DetectionScene

        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        detection_scene = DetectionScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(BoundingBox, Label),  # pyrefly: ignore[bad-argument-type]
            scale=scale,
        )
        rendered = detection_scene.render()
        return np.array(rendered)


class InstanceSegmentationVisualizerCreator(VisualizerCreator):
    """Creator for instance segmentation visualizations."""

    def create_visualization(  # pyrefly: ignore[bad-override]
        self,
        original_image: np.ndarray,
        predictions: "InstanceSegmentationResult",
    ) -> np.ndarray:
        from model_api.models.result import Label
        from model_api.visualizer import Flatten, Polygon
        from model_api.visualizer.scene import InstanceSegmentationScene

        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        segmentation_scene = InstanceSegmentationScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(Polygon, Label),  # pyrefly: ignore[bad-argument-type]
            scale=scale,
        )
        rendered = segmentation_scene.render()
        return np.array(rendered)


class VisualizationDispatcher(metaclass=Singleton):
    """Dispatcher for creating visualizations."""

    def __init__(self) -> None:
        from model_api.models.result import ClassificationResult, DetectionResult, InstanceSegmentationResult, Result

        self._creator_map: dict[type[Result], VisualizerCreator] = {
            DetectionResult: DetectionVisualizerCreator(),
            ClassificationResult: ClassificationVisualizerCreator(),
            InstanceSegmentationResult: InstanceSegmentationVisualizerCreator(),
        }

    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: "Result",
    ) -> np.ndarray | None:
        if original_image.size == 0:
            raise ValueError("The image provided through the 'original_image' parameter cannot be empty.")

        creator = self._creator_map.get(type(predictions))
        if creator is not None:
            return creator.create_visualization(original_image, predictions)
        logger.error("Visualization for {} is not supported.", type(predictions))
        return None


class Visualizer:
    @staticmethod
    def overlay_predictions(
        original_image: np.ndarray,
        predictions: "Result",
    ) -> np.ndarray:
        """Overlay predictions on the original image.

        Args:
            original_image: BGR/RGB numpy image.
            predictions: Model API prediction result.
        """
        try:
            visualization = VisualizationDispatcher().create_visualization(original_image, predictions)
            if visualization is None:
                return original_image
        except Exception:
            logger.exception("An error occurred while creating visualization, returning original image.")
            return original_image
        return visualization
