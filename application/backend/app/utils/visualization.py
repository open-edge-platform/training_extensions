# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from loguru import logger
from model_api.models import ClassificationResult
from model_api.models.result import DetectionResult, InstanceSegmentationResult, Label, Result
from model_api.visualizer import BoundingBox, Flatten, Polygon
from model_api.visualizer.defaults import SCALE_BASELINE
from model_api.visualizer.primitive import Label as LabelPrimitive
from model_api.visualizer.primitive import Polygon as PolygonPrimitive
from model_api.visualizer.scene import ClassificationScene, DetectionScene, InstanceSegmentationScene
from model_api.visualizer.scene.scene import Scene
from PIL import Image

from app.utils.singleton import Singleton

R = TypeVar("R", bound=Result)


def _compute_scale(image: np.ndarray) -> float:
    """Compute a font/outline scale factor based on the image's longer edge.

    Uses SCALE_BASELINE (720p longer edge = 1280) as the reference: at 1280px the
    scale is 1.0; at 4K (3840px) the scale is ~3.0. Never shrinks below 1.0.
    """
    if image is None or image.size == 0:
        return 1.0
    h, w = image.shape[:2]
    longer_edge = float(max(int(h), int(w)))
    return max(1.0, longer_edge / float(SCALE_BASELINE))


def _hex_to_rgb(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(color, tuple):
        return color
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _contrasting_fg(bg_color: str | tuple[int, int, int]) -> str:
    """Return 'white' or 'black' depending on which gives better contrast on bg_color."""
    r, g, b = _hex_to_rgb(bg_color)
    # Relative luminance (sRGB approximation).
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "black" if luminance > 0.6 else "white"


def _normalize_label_name(name: str) -> str:
    """Normalize a label name for matching.

    During model export, label names have spaces replaced with underscores
    (see ``TaskLevelExportParameters.to_metadata``), so a prediction's label
    name won't be byte-equal to the UI/DB name. We canonicalize both sides by
    lower-casing and replacing spaces with underscores.
    """
    return name.strip().lower().replace(" ", "_")


def _build_normalized_color_map(label_colors: dict[str, str]) -> dict[str, str]:
    """Return ``{normalized_name: hex_color}`` so we can match model_api's munged names."""
    return {_normalize_label_name(name): color for name, color in label_colors.items()}


def _build_normalized_display_map(label_colors: dict[str, str]) -> dict[str, str]:
    """Return ``{normalized_name: original_display_name}`` for label-text restoration."""
    return {_normalize_label_name(name): name for name in label_colors}


def _lookup_color(name: str, normalized_colors: dict[str, str]) -> str | None:
    return normalized_colors.get(_normalize_label_name(name))


def _lookup_display_name(name: str, normalized_display: dict[str, str]) -> str | None:
    return normalized_display.get(_normalize_label_name(name))


def _split_label_and_score(label_text: str) -> tuple[str, str]:
    """Return ``(name, score_suffix)`` where ``score_suffix`` is the trailing ' (0.xx)' or ''."""
    if label_text.endswith(")") and " (" in label_text:
        name, _, score = label_text.rpartition(" (")
        return name, f" ({score}"
    return label_text, ""


def _recolor_bounding_boxes(
    scene: "Scene",
    normalized_colors: dict[str, str],
    normalized_display: dict[str, str],
) -> None:
    for bbox in scene.get_primitives(BoundingBox):
        label_text = getattr(bbox, "label", None)
        if not isinstance(label_text, str):
            continue
        name, score_suffix = _split_label_and_score(label_text)
        color = _lookup_color(name, normalized_colors)
        if color is not None:
            bbox.color = color  # type: ignore[attr-defined]
        display = _lookup_display_name(name, normalized_display)
        if display is not None:
            bbox.label = f"{display}{score_suffix}"  # type: ignore[attr-defined]


def _recolor_polygons(scene: "Scene", normalized_colors: dict[str, str]) -> None:
    # model_api doesn't store the label name on Polygon, so we match by current
    # color via the scene's color_per_label map (populated by Detection/InstSeg
    # scenes via ``get_label_color_mapping``).
    color_per_label = getattr(scene, "color_per_label", None)
    if not color_per_label:
        return
    upstream_to_name: dict[str, str] = {v: k for k, v in color_per_label.items()}
    for poly in scene.get_primitives(PolygonPrimitive):
        current = getattr(poly, "color", None)
        if not isinstance(current, str):
            continue
        name = upstream_to_name.get(current)
        if name is None:
            continue
        color = _lookup_color(name, normalized_colors)
        if color is not None:
            poly.color = color  # type: ignore[attr-defined]
    # Keep the scene's mapping consistent for any downstream lookups.
    for name in list(color_per_label.keys()):
        color = _lookup_color(name, normalized_colors)
        if color is not None:
            color_per_label[name] = color


def _recolor_labels(
    scene: "Scene",
    normalized_colors: dict[str, str],
    normalized_display: dict[str, str],
) -> None:
    for lbl in scene.get_primitives(LabelPrimitive):
        label_text = getattr(lbl, "label", None)
        if not isinstance(label_text, str):
            continue
        name, score_suffix = _split_label_and_score(label_text)
        color = _lookup_color(name, normalized_colors)
        if color is not None:
            lbl.bg_color = color  # type: ignore[attr-defined]
            lbl.fg_color = _contrasting_fg(color)  # type: ignore[attr-defined]
        display = _lookup_display_name(name, normalized_display)
        if display is not None:
            lbl.label = f"{display}{score_suffix}"  # type: ignore[attr-defined]


def _recolor_primitives(scene: "Scene", label_colors: dict[str, str] | None) -> None:
    """Override colors and label text on the scene's primitives using the provided name->hex map.

    Label names are matched case-insensitively against ``label_colors`` after
    normalizing non-alphanumeric runs to ``_`` (since ``model_api`` rewrites
    names to be identifier-friendly). The original display name from
    ``label_colors`` is then restored on each primitive's ``.label``.
    """
    if not label_colors:
        return
    normalized_colors = _build_normalized_color_map(label_colors)
    normalized_display = _build_normalized_display_map(label_colors)
    _recolor_bounding_boxes(scene, normalized_colors, normalized_display)
    _recolor_polygons(scene, normalized_colors)
    _recolor_labels(scene, normalized_colors, normalized_display)


class VisualizerCreator(ABC, Generic[R]):
    """Abstract base class for visualizer creators."""

    @abstractmethod
    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: R,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Create a visualization of the predictions on the original image."""


class ClassificationVisualizerCreator(VisualizerCreator[ClassificationResult]):
    """Creator for classification visualizations."""

    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: ClassificationResult,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray:
        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        classification_scene = ClassificationScene(
            image=image_pil,
            result=predictions,
            scale=scale,
        )
        _recolor_primitives(classification_scene, label_colors)
        rendered = classification_scene.render()
        return np.array(rendered)


class DetectionVisualizerCreator(VisualizerCreator[DetectionResult]):
    """Creator for detection visualizations."""

    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: DetectionResult,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray:
        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        detection_scene = DetectionScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(BoundingBox, Label),  # pyrefly: ignore[bad-argument-type]
            scale=scale,
        )
        _recolor_primitives(detection_scene, label_colors)
        rendered = detection_scene.render()
        return np.array(rendered)


class InstanceSegmentationVisualizerCreator(VisualizerCreator[InstanceSegmentationResult]):
    """Creator for instance segmentation visualizations."""

    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: InstanceSegmentationResult,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray:
        image_pil = Image.fromarray(original_image)
        scale = _compute_scale(original_image)
        segmentation_scene = InstanceSegmentationScene(
            image=image_pil,
            result=predictions,
            layout=Flatten(Polygon, Label),  # pyrefly: ignore[bad-argument-type]
            scale=scale,
        )
        _recolor_primitives(segmentation_scene, label_colors)
        rendered = segmentation_scene.render()
        return np.array(rendered)


class VisualizationDispatcher(metaclass=Singleton):
    """Dispatcher for creating visualizations."""

    def __init__(self) -> None:
        self._creator_map: dict[type[Result], VisualizerCreator] = {
            DetectionResult: DetectionVisualizerCreator(),
            ClassificationResult: ClassificationVisualizerCreator(),
            InstanceSegmentationResult: InstanceSegmentationVisualizerCreator(),
        }

    def create_visualization(
        self,
        original_image: np.ndarray,
        predictions: Result,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray | None:
        if original_image.size == 0:
            raise ValueError("The image provided through the 'original_image' parameter cannot be empty.")

        creator = self._creator_map.get(type(predictions))
        if creator is not None:
            return creator.create_visualization(original_image, predictions, label_colors=label_colors)
        logger.error("Visualization for {} is not supported.", type(predictions))
        return None


class Visualizer:
    @staticmethod
    def overlay_predictions(
        original_image: np.ndarray,
        predictions: Result,
        label_colors: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Overlay predictions on the original image.

        Args:
            original_image: BGR/RGB numpy image.
            predictions: Model API prediction result.
            label_colors: Optional mapping from label name -> hex color (as configured
                in the UI). When provided, primitive colors are overridden so the
                rendered overlay matches the UI's annotation palette.
        """
        try:
            visualization = VisualizationDispatcher().create_visualization(
                original_image, predictions, label_colors=label_colors
            )
            if visualization is None:
                return original_image
        except Exception:
            logger.exception("An error occurred while creating visualization, returning original image.")
            return original_image
        return visualization
