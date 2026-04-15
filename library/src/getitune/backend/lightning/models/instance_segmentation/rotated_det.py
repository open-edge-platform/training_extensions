# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rotated Detection Prediction Mixin."""

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

<<<<<<<< HEAD:library/src/getitune/backend/lightning/models/instance_segmentation/rotated_det.py
from getitune.data.entity.sample import PredictionBatch
========
from getitune.data.entity.sample import OTXPredictionBatch
>>>>>>>> develop:library/src/getitune/backend/native/models/instance_segmentation/rotated_det.py


def get_polygon_area(points: np.ndarray) -> float:
    """Calculate polygon area using the shoelace formula.

    Args:
        points: Array of polygon vertices with shape (N, 2)

    Returns:
        float: Area of the polygon
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convert_masks_to_rotated_predictions(preds: PredictionBatch) -> PredictionBatch:
    """Convert masks to rotated bounding boxes.

    This function processes the predictions from an instance segmentation model,
    extracting rotated bounding boxes from the masks.

    Args:
        preds (PredictionBatch): The predictions from the instance segmentation model.

    Returns:
        PredictionBatch: The predictions with rotated bounding boxes.
    """
    batch_scores = []
    batch_bboxes = []
    batch_labels = []
    batch_masks = []

    for field_name, field in zip(
        ["imgs_info", "bboxes", "scores", "labels", "masks"],
        [preds.imgs_info, preds.bboxes, preds.scores, preds.labels, preds.masks],
    ):
        if field is None:
            msg = f"Field '{field_name}' is None, which is not allowed."
            raise ValueError(msg)

    for img_info, pred_bboxes, pred_scores, pred_labels, pred_masks in zip(  # type: ignore[misc]
        preds.imgs_info,  # type: ignore[arg-type]
        preds.bboxes,  # type: ignore[arg-type]
        preds.scores,  # type: ignore[arg-type]
        preds.labels,  # type: ignore[arg-type]
        preds.masks,  # type: ignore[arg-type]
    ):
        boxes, scores, labels, masks = [], [], [], []

        for bbox, score, label, mask in zip(pred_bboxes, pred_scores, pred_labels, pred_masks):
            if mask.sum() == 0:
                continue
            np_mask = mask.detach().cpu().numpy().astype(int)
            contours, hierarchies = cv2.findContours(np_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchies is None:
                continue

            # Find the largest contour for the rotated bounding box
            valid_contours = []
            for contour, hierarchy in zip(contours, hierarchies[0]):
                if hierarchy[3] != -1 or len(contour) <= 2:
                    continue
                box_points = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
                area = get_polygon_area(box_points)
                valid_contours.append((box_points, area))

            if valid_contours:
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                scores.append(score)
                boxes.append(bbox)
                labels.append(label)
                masks.append(mask)

        if boxes:
            scores = torch.stack(scores)
            boxes = tv_tensors.BoundingBoxes(torch.stack(boxes), format="XYXY", canvas_size=img_info.ori_shape)  # type: ignore[union-attr]
            labels = torch.stack(labels)
            masks = torch.stack(masks)

        batch_scores.append(scores)
        batch_bboxes.append(boxes)
        batch_labels.append(labels)
        batch_masks.append(masks)

    return PredictionBatch(
        images=preds.images,
        imgs_info=preds.imgs_info,
        scores=batch_scores,
        bboxes=batch_bboxes,
        masks=batch_masks,
        labels=batch_labels,
    )


class RotatedPredictMixin:
    """Mixin class for rotated detection prediction."""

    def rotated_predict_step(self, preds: PredictionBatch) -> PredictionBatch:
        """Perform prediction step for rotated detection."""
        return convert_masks_to_rotated_predictions(preds)
