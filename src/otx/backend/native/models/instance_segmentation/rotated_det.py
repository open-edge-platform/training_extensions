# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rotated Detection Prediction Mixin."""

from otx.data.entity.torch.torch import OTXPredBatch

from datumaro import Polygon
import cv2
import torch
from torchvision import tv_tensors


def convert_masks_to_rotated_predictions(preds: OTXPredBatch) -> OTXPredBatch:
    """Convert masks to rotated bounding boxes and polygons.
    This function processes the predictions from an instance segmentation model,
    extracting rotated bounding boxes and polygons from the masks.

    Args:
        preds (OTXPredBatch): The predictions from the instance segmentation model.

    Returns:
        OTXPredBatch: The predictions with rotated bounding boxes and polygons.
    """
    batch_scores = []
    batch_bboxes = []
    batch_labels = []
    batch_polygons = []
    batch_masks = []

    for field_name, field in zip(
        ["imgs_info", "bboxes", "scores", "labels", "masks"],
        [preds.imgs_info, preds.bboxes, preds.scores, preds.labels, preds.masks],
    ):
        if field is None:
            raise ValueError(f"Field '{field_name}' is None, which is not allowed.")

    for img_info, pred_bboxes, pred_scores, pred_labels, pred_masks in zip(
        preds.imgs_info, preds.bboxes, preds.scores, preds.labels, preds.masks  # type: ignore[arg-type,misc]
    ):
        boxes, scores, labels, masks, polygons = [], [], [], [], []

        for bbox, score, label, mask in zip(pred_bboxes, pred_scores, pred_labels, pred_masks):
            if mask.sum() == 0:
                continue
            np_mask = mask.detach().cpu().numpy().astype(int)
            contours, hierarchies = cv2.findContours(np_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchies is None:
                continue

            rbox_polygons = []
            for contour, hierarchy in zip(contours, hierarchies[0]):
                if hierarchy[3] != -1 or len(contour) <= 2:
                    continue
                rbox_points = Polygon(cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1))
                rbox_polygons.append((rbox_points, rbox_points.get_area()))

            if rbox_polygons:
                rbox_polygons.sort(key=lambda x: x[1], reverse=True)
                polygons.append(rbox_polygons[0][0])
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
        batch_polygons.append(polygons)
        batch_masks.append(masks)

    return OTXPredBatch(
        batch_size=preds.batch_size,
        images=preds.images,
        imgs_info=preds.imgs_info,
        scores=batch_scores,
        bboxes=batch_bboxes,
        masks=batch_masks,
        polygons=batch_polygons,
        labels=batch_labels,
    )


class RotatedPredictMixin:
    """Mixin class for rotated detection prediction."""
    def rotated_predict_step(self, preds: OTXPredBatch) -> OTXPredBatch:
        """Perform prediction step for rotated detection."""
        return convert_masks_to_rotated_predictions(preds)