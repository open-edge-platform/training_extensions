# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from loguru import logger
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult
from model_api.models.result import Result

from app.models import DatasetItemAnnotation, FullImage, Label, LabelReference, Point, Polygon, Rectangle


def _build_label_maps(labels: Sequence[Label]) -> tuple[dict[str, Label], dict[str, Label]]:
    """Precompute name→label and unmangled-name→label dicts for O(1) lookup."""
    name_map = {label.name: label for label in labels}
    unmangled_map = {label.name.replace(" ", "_"): label for label in labels if " " in label.name}
    return name_map, unmangled_map


def _find_project_label(
    name_map: dict[str, Label],
    unmangled_map: dict[str, Label],
    labels: Sequence[Label],
    label_name: str | None,
    predicted_class_index: int | None = None,
) -> Label | None:
    """Find a project label matching the predicted label using a three-stage fallback strategy.

    Resolution order:
    1. **Exact name match** - performs an O(1) lookup of ``label_name`` in ``name_map``.
    2. **Unmangled name match** - if the exact match fails and ``label_name`` contains underscores, performs an O(1)
       lookup in ``unmangled_map``. This handles models that serialize label names with underscores instead of spaces
       (e.g. ``"my_cat"`` → ``"my cat"``). The lookup is skipped entirely when ``label_name`` contains no underscores.
    3. **Index fallback** - if both name lookups fail, returns ``labels[predicted_class_index]`` when the index is a
       non-negative integer within the bounds of ``labels``. This handles renamed labels whose in-model
       name no longer matches any project label.

    Args:
        name_map: Mapping of label name → ``Label`` for O(1) exact-name lookup.
        unmangled_map: Mapping of underscore-mangled label name → ``Label`` for O(1) unmangled-name lookup.
            Only contains entries for labels whose names include spaces.
        labels: Ordered sequence of project labels to search.
        label_name: The label name as reported by the model. May be ``None`` when the model does not provide a name,
            in which case stages 1 and 2 are skipped.
        predicted_class_index: Zero-based class index produced by the model. Used only when name-based lookup fails.
            Defaults to ``None``, which disables the index fallback.

    Returns:
        The matched ``Label``, or ``None`` if no stage produced a match.
    """
    if label_name is not None:
        label = name_map.get(label_name)
        if label:
            return label

        # Only attempt unmangled lookup if "_" is present (avoids unnecessary dict lookup noise)
        if "_" in label_name:
            label = unmangled_map.get(label_name)
            if label:
                return label

    if predicted_class_index is not None and 0 <= predicted_class_index < len(labels):
        return labels[predicted_class_index]
    return None


def _convert_classification_prediction(
    labels: Sequence[Label], prediction: ClassificationResult
) -> list[DatasetItemAnnotation]:
    predicted_labels: list[LabelReference] = []
    predicted_confidences: list[float] = []
    if prediction.top_labels is None:
        raise RuntimeError("The prediction is malformed because it does not contain labels")
    name_map, unmangled_map = _build_label_maps(labels)
    for predicted_label in prediction.top_labels:
        label_name = predicted_label.name
        label = _find_project_label(
            name_map=name_map,
            unmangled_map=unmangled_map,
            labels=labels,
            label_name=label_name,
            predicted_class_index=predicted_label.id,
        )
        if not label:
            logger.warning("Prediction label {} cannot be found in the project", label_name)
            continue
        confidence = predicted_label.confidence
        if confidence is None:
            logger.warning("The predicted label {} does not have a confidence score; assuming 1.0", label_name)
            confidence = 1.0
        predicted_labels.append(LabelReference(id=label.id))
        predicted_confidences.append(confidence)
    return [DatasetItemAnnotation(labels=predicted_labels, shape=FullImage(), confidences=predicted_confidences)]


def _convert_detection_prediction(labels: Sequence[Label], prediction: DetectionResult) -> list[DatasetItemAnnotation]:
    name_map, unmangled_map = _build_label_maps(labels)
    result = []
    prediction_scores_list = prediction.scores.tolist()
    for idx, box in enumerate(prediction.bboxes):
        label_name = prediction.label_names[idx]
        bbox_confidence = prediction_scores_list[idx]
        label = _find_project_label(
            name_map=name_map,
            unmangled_map=unmangled_map,
            labels=labels,
            label_name=label_name,
            predicted_class_index=int(prediction.labels[idx]),
        )
        if not label:
            logger.warning("Prediction label {} cannot be found in the project", label_name)
            continue
        x1, y1, x2, y2 = box.tolist()
        annotation = DatasetItemAnnotation(
            labels=[LabelReference(id=label.id)],
            shape=Rectangle(x=x1, y=y1, width=(x2 - x1), height=(y2 - y1)),
            confidences=[bbox_confidence],
        )
        result.append(annotation)
    return result


def _convert_segmentation_prediction(
    labels: Sequence[Label],
    frame_data: np.ndarray,
    prediction: InstanceSegmentationResult,
) -> list[DatasetItemAnnotation]:
    name_map, unmangled_map = _build_label_maps(labels)
    height, width, _ = frame_data.shape
    result = []
    prediction_scores_list = prediction.scores.tolist()
    for idx, box in enumerate(prediction.bboxes):
        label_name = prediction.label_names[idx]
        polygon_confidence = prediction_scores_list[idx]
        label = _find_project_label(
            name_map=name_map,
            unmangled_map=unmangled_map,
            labels=labels,
            label_name=label_name,
            predicted_class_index=int(prediction.labels[idx]),
        )
        if not label:
            logger.warning("Prediction label {} cannot be found in the project", label_name)
            continue
        mask = prediction.masks[idx].astype(np.uint8)
        contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchies is None:
            continue
        for contour, hierarchy in zip(contours, hierarchies[0]):
            if hierarchy[3] != -1:
                continue
            if len(contour) <= 2 or cv2.contourArea(contour) < 1.0:
                continue
            polygon = Polygon(points=[Point(x=point[0][0], y=point[0][1]) for point in list(contour)])
            annotation = DatasetItemAnnotation(
                labels=[LabelReference(id=label.id)], shape=polygon, confidences=[polygon_confidence]
            )
            result.append(annotation)
    return result


def get_confidence_scores(prediction: Result) -> list[float]:
    """
    Gets model prediction confidence scores depending on the
    specific type of prediction result (segmentation, detection, or classification).

    Args:
        prediction: Prediction result object containing model outputs, which can
                   be one of: InstanceSegmentationResult, DetectionResult, or
                   ClassificationResult.

    Returns:
        list[float]: List of confidence scores.
    """
    match prediction:
        case InstanceSegmentationResult() | DetectionResult():
            return prediction.scores.tolist()
        case ClassificationResult():
            if prediction.top_labels is None:
                raise RuntimeError("The prediction is malformed because it does not contain labels")
            return [cast(float, label.confidence) for label in prediction.top_labels]
    return []


def convert_prediction(
    labels: Sequence[Label], frame_data: np.ndarray, prediction: Result
) -> list[DatasetItemAnnotation]:
    """
    Converts model predictions to dataset annotations based on prediction type.

    Routes the conversion process to appropriate handlers depending on the
    specific type of prediction result (segmentation, detection, or classification).

    Args:
        labels: List of Label objects available in the project for annotation.
        frame_data: Image data in numpy ndarray format, used for segmentation
                   annotations that may require image dimensions.
        prediction: Prediction result object containing model outputs, which can
                   be one of: InstanceSegmentationResult, DetectionResult, or
                   ClassificationResult.

    Returns:
        list[DatasetItemAnnotation]: List of annotations converted from the
        prediction results. Returns empty list if prediction type is not recognized.

    Note:
        The function uses pattern matching to dispatch to appropriate conversion
        methods based on the prediction type. Each prediction type has its own
        specialized conversion logic.

    Example:
        >>> annotations = convert_prediction(
        ...     labels=project_labels,
        ...     frame_data=image_array,
        ...     prediction=detection_result
        ... )
    """
    match prediction:
        case InstanceSegmentationResult():
            return _convert_segmentation_prediction(labels=labels, frame_data=frame_data, prediction=prediction)
        case DetectionResult():
            return _convert_detection_prediction(labels=labels, prediction=prediction)
        case ClassificationResult():
            return _convert_classification_prediction(labels=labels, prediction=prediction)
    return []
