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


def _convert_classification_prediction(
    labels: Sequence[Label], prediction: ClassificationResult
) -> list[DatasetItemAnnotation]:
    predicted_labels: list[LabelReference] = []
    predicted_confidences: list[float] = []
    if prediction.top_labels is None:
        raise RuntimeError("The prediction is malformed because it does not contain labels")
    for predicted_label in prediction.top_labels:
        label_name = predicted_label.name
        label = next((label for label in labels if label.name == label_name), None)
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
    result = []
    prediction_scores_list = prediction.scores.tolist()
    for idx, box in enumerate(prediction.bboxes):
        label_name = prediction.label_names[idx]
        bbox_confidence = prediction_scores_list[idx]
        label = next((label for label in labels if label.name == label_name), None)
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
    height, width, _ = frame_data.shape
    result = []
    prediction_scores_list = prediction.scores.tolist()
    for idx, box in enumerate(prediction.bboxes):
        label_name = prediction.label_names[idx]
        polygon_confidence = prediction_scores_list[idx]
        label = next((label for label in labels if label.name == label_name), None)
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
