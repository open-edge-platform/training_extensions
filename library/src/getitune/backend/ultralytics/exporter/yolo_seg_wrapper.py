# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom ModelAPI wrapper for YOLO instance-segmentation inference.

YOLO segmentation models export with ``end2end=False`` produce two outputs:
  * Output 0: ``[1, num_classes + 4 + mask_dim, num_boxes]`` — detection
    predictions concatenated with mask coefficients.
  * Output 1: ``[1, mask_dim, proto_h, proto_w]`` — mask prototypes.

The standard ModelAPI ``YOLO11`` wrapper only handles a single detection
output. This wrapper extends it to decode instance masks by multiplying
mask coefficients by prototypes, cropping to bounding boxes, and resizing
to original image dimensions.

The wrapper registers itself as ``"YOLO11-seg"`` so OVEngine can discover
it via ``model_type`` metadata in the exported IR.
"""

from __future__ import annotations

import numpy as np
from model_api.models.detection_model import DetectionModel
from model_api.models.utils import InstanceSegmentationResult, ResizeMetadata


def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2] in-place."""
    return np.stack(
        (
            xywh[:, 0] - xywh[:, 2] / 2.0,
            xywh[:, 1] - xywh[:, 3] / 2.0,
            xywh[:, 0] + xywh[:, 2] / 2.0,
            xywh[:, 1] + xywh[:, 3] / 2.0,
        ),
        axis=1,
        out=xywh,
    )


def _crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zero-out mask pixels outside the bounding box.

    Args:
        masks: Binary or float masks of shape ``(N, H, W)``.
        boxes: Bounding boxes ``(N, 4)`` in xyxy format, scaled to mask dims.

    Returns:
        Cropped masks of shape ``(N, H, W)``.
    """
    n, h, w = masks.shape
    # Create coordinate grids
    rows = np.arange(h, dtype=np.float32).reshape(1, h, 1)
    cols = np.arange(w, dtype=np.float32).reshape(1, 1, w)
    # boxes: (N, 4) -> separate x1, y1, x2, y2
    x1 = boxes[:, 0].reshape(n, 1, 1)
    y1 = boxes[:, 1].reshape(n, 1, 1)
    x2 = boxes[:, 2].reshape(n, 1, 1)
    y2 = boxes[:, 3].reshape(n, 1, 1)
    # Mask pixels inside the box
    inside = (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)
    return masks * inside


class YOLO11Seg(DetectionModel):
    """ModelAPI wrapper for YOLO11 instance-segmentation models.

    Expects 2 outputs:
      * detection output: ``[1, 4 + num_classes + mask_dim, num_boxes]``
      * prototype output: ``[1, mask_dim, proto_h, proto_w]``

    Post-processing:
      1. Parse detection predictions (boxes + class scores + mask coefficients).
      2. Filter by confidence, apply NMS.
      3. Decode masks: ``coefficients @ protos.reshape(mask_dim, -1)`` → sigmoid → crop → resize.
      4. Return ``InstanceSegmentationResult``.
    """

    __model__ = "YOLO11-seg"

    def __init__(self, inference_adapter: object, configuration: dict | None = None, preload: bool = False) -> None:
        super().__init__(inference_adapter, configuration or {}, preload)
        self._check_io_number(1, 2)

        # Identify outputs by shape: detection is rank-3, prototypes is rank-4
        self._det_output_name: str = ""
        self._proto_output_name: str = ""

        for name, output in self.outputs.items():
            shape = output.shape
            if len(shape) == 3:
                self._det_output_name = name
            elif len(shape) == 4:
                self._proto_output_name = name

        if not self._det_output_name or not self._proto_output_name:
            self.raise_error(
                "Expected one rank-3 detection output and one rank-4 prototype output, "
                f"but got shapes: {[(name, out.shape) for name, out in self.outputs.items()]}"
            )

        det_shape = self.outputs[self._det_output_name].shape
        proto_shape = self.outputs[self._proto_output_name].shape
        self._mask_dim = proto_shape[1]  # typically 32
        self._proto_h = proto_shape[2]
        self._proto_w = proto_shape[3]

        # Validate detection output structure: [1, 4 + num_classes + mask_dim, num_boxes]
        self._num_classes = det_shape[1] - 4 - self._mask_dim
        if self._num_classes <= 0:
            self.raise_error(f"Detection output channel dim ({det_shape[1]}) must be > 4 + mask_dim ({self._mask_dim})")

    @classmethod
    def parameters(cls) -> dict:
        """Default parameter values for YOLO11-seg inference."""
        parameters = super().parameters()
        parameters["pad_value"].update_default_value(114)
        parameters["resize_type"].update_default_value("fit_to_window_letterbox")
        parameters["reverse_input_channels"].update_default_value(True)
        parameters["scale_values"].update_default_value([255.0])
        parameters["confidence_threshold"].update_default_value(0.25)
        parameters["nms_execute"].update_default_value(default_value=True)
        parameters["iou_threshold"].update_default_value(0.7)
        parameters["nms_max_predictions"].update_default_value(30000)
        return parameters

    def postprocess(self, outputs: dict, meta: dict) -> InstanceSegmentationResult:
        """Decode detections and instance masks from raw model outputs.

        Args:
            outputs: Raw model outputs keyed by output tensor name.
            meta: Preprocessing metadata from ModelAPI (original_shape, etc.).

        Returns:
            InstanceSegmentationResult with boxes in original image coordinates
            and binary masks at original image resolution.
        """
        det_output = outputs[self._det_output_name]  # [1, C, N]
        proto_output = outputs[self._proto_output_name]  # [1, mask_dim, proto_h, proto_w]

        prediction = det_output.astype(np.float32)
        protos = proto_output[0].astype(np.float32)  # [mask_dim, proto_h, proto_w]

        # Transpose to [N, C] for easier processing
        pred = prediction[0].T  # [num_boxes, 4 + num_classes + mask_dim]

        # Split into bbox, class scores, mask coefficients
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4 : 4 + self._num_classes]
        mask_coeffs = pred[:, 4 + self._num_classes :]  # [N, mask_dim]

        # Filter by confidence
        conf_threshold = self.params.confidence_threshold
        max_scores = class_scores.max(axis=1)
        keep_conf = max_scores > conf_threshold

        if not keep_conf.any():
            return self._empty_result(meta)

        boxes_xywh = boxes_xywh[keep_conf]
        class_scores = class_scores[keep_conf]
        mask_coeffs = mask_coeffs[keep_conf]

        # Get class labels and confidences
        labels = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(labels)), labels]

        # Convert to xyxy
        boxes_xyxy = _xywh2xyxy(boxes_xywh.copy())

        # NMS
        keep_nms = self._calculate_nms(
            boxes=boxes_xyxy,
            scores=confidences,
            labels=labels.astype(np.float32),
        )
        boxes_xyxy = boxes_xyxy[keep_nms]
        confidences = confidences[keep_nms]
        labels = labels[keep_nms]
        mask_coeffs = mask_coeffs[keep_nms]

        if len(boxes_xyxy) == 0:
            return self._empty_result(meta)

        # Decode masks: coefficients @ protos -> sigmoid -> crop -> resize
        masks = self._decode_masks(mask_coeffs, protos, boxes_xyxy, meta)

        # Scale boxes to original image coordinates
        input_img_w = meta["original_shape"][1]
        input_img_h = meta["original_shape"][0]
        resize_meta = ResizeMetadata.compute(
            original_width=input_img_w,
            original_height=input_img_h,
            model_width=self.orig_width,
            model_height=self.orig_height,
            resize_type=self.params.resize_type,
        )

        coords = boxes_xyxy.copy()
        coords -= (resize_meta.pad_left, resize_meta.pad_top, resize_meta.pad_left, resize_meta.pad_top)
        coords *= (
            resize_meta.inverted_scale_x,
            resize_meta.inverted_scale_y,
            resize_meta.inverted_scale_x,
            resize_meta.inverted_scale_y,
        )

        int_boxes = np.round(coords).astype(np.int32)
        np.clip(
            int_boxes,
            0,
            [input_img_w, input_img_h, input_img_w, input_img_h],
            out=int_boxes,
        )

        int_labels = labels.astype(np.int32)
        return InstanceSegmentationResult(
            bboxes=int_boxes,
            scores=confidences,
            labels=int_labels + 1,  # 1-indexed to match MaskRCNN/ModelAPI convention
            masks=masks,
            label_names=[self.get_label_name(i) for i in int_labels],
            saliency_map=[],
            feature_vector=np.ndarray(0),
        )

    def _decode_masks(
        self,
        mask_coeffs: np.ndarray,
        protos: np.ndarray,
        boxes_xyxy: np.ndarray,
        meta: dict,
    ) -> np.ndarray:
        """Decode instance masks from mask coefficients and prototypes.

        Args:
            mask_coeffs: Mask coefficients ``(N, mask_dim)``.
            protos: Prototype masks ``(mask_dim, proto_h, proto_w)``.
            boxes_xyxy: Bounding boxes in model input coordinates ``(N, 4)``.
            meta: Preprocessing metadata (original_shape, etc.).

        Returns:
            Binary masks at original image resolution ``(N, orig_h, orig_w)``.
        """
        # masks = coefficients @ protos.reshape(mask_dim, -1) -> reshape to (N, proto_h, proto_w)
        mask_dim, proto_h, proto_w = protos.shape
        raw_masks = mask_coeffs @ protos.reshape(mask_dim, -1)  # (N, proto_h * proto_w)
        raw_masks = raw_masks.reshape(-1, proto_h, proto_w)  # (N, proto_h, proto_w)

        # Sigmoid activation
        raw_masks = 1.0 / (1.0 + np.exp(-raw_masks))

        # Scale boxes to prototype dimensions for cropping
        model_h, model_w = self.orig_height, self.orig_width
        scale_x = proto_w / model_w
        scale_y = proto_h / model_h
        proto_boxes = boxes_xyxy * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)

        # Crop masks to bounding box regions
        raw_masks = _crop_mask(raw_masks, proto_boxes)

        # Resize masks to original image dimensions
        input_img_h = meta["original_shape"][0]
        input_img_w = meta["original_shape"][1]

        # First resize to model input size, removing padding
        resize_meta = ResizeMetadata.compute(
            original_width=input_img_w,
            original_height=input_img_h,
            model_width=model_w,
            model_height=model_h,
            resize_type=self.params.resize_type,
        )

        # Upsample from proto resolution to model input size
        from model_api.adapters.utils import resize_image_ocv

        n = raw_masks.shape[0]
        upsampled = np.zeros((n, model_h, model_w), dtype=np.float32)
        for i in range(n):
            upsampled[i] = resize_image_ocv(raw_masks[i], (model_w, model_h))

        # Remove padding and scale to original image size
        pad_t = resize_meta.pad_top
        pad_l = resize_meta.pad_left
        # Compute effective (non-padded) dimensions from the scale factor
        effective_w = round(input_img_w / resize_meta.inverted_scale_x)
        effective_h = round(input_img_h / resize_meta.inverted_scale_y)
        cropped = upsampled[:, pad_t : pad_t + effective_h, pad_l : pad_l + effective_w]

        # Resize to original image size
        final_masks = np.zeros((n, input_img_h, input_img_w), dtype=np.uint8)
        for i in range(n):
            resized = resize_image_ocv(cropped[i], (input_img_w, input_img_h))
            final_masks[i] = (resized > 0.5).astype(np.uint8)

        return final_masks

    def _empty_result(self, meta: dict) -> InstanceSegmentationResult:
        """Return an empty result when no detections pass filtering."""
        return InstanceSegmentationResult(
            bboxes=np.empty((0, 4), dtype=np.int32),
            scores=np.empty(0, dtype=np.float32),
            labels=np.empty(0, dtype=np.int32),
            masks=np.empty((0, meta["original_shape"][0], meta["original_shape"][1]), dtype=np.uint8),
            label_names=[],
            saliency_map=[],
            feature_vector=np.ndarray(0),
        )
