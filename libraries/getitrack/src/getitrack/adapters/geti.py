# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Adapter for getitune detection models.

getitune and torch are imported lazily inside the methods that need them, so
getitrack imports without either installed. `GetiAdapter.to_detections` is
duck-typed and works on any prediction-batch-shaped object.
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from getitrack.adapters.base import DetectionAdapter
from getitrack.core.detection import Detections

# Matches getitune placeholder class names like "label_0", "label_1".
_PLACEHOLDER_NAME = re.compile(r"^label_\d+$")


class GetiAdapter(DetectionAdapter):
    """Runs a getitune detection model on raw BGR frames.

    Wraps the preprocess, ``predict_step``, and postprocess round trip
    of a getitune Lightning detection model (RF-DETR, YOLOX, ...) for
    inference outside the getitune Trainer.

    Example:
        >>> adapter = GetiAdapter(model, device="cuda")
        >>> detections = adapter.detect(frame, frame_id=0)
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:  # noqa: ANN401
        """Wrap a getitune detection model.

        Args:
            model: getitune detection model in eval mode, exposing
                ``data_input_params``, ``predict_step``, and ``label_info``.
            device: Torch device the model lives on.
        """
        self.model = model
        self.device = device

    @property
    def class_names(self) -> dict[int, str] | None:
        """Class names read from the model's ``label_info``.

        getitune models trained or loaded against a dataset carry real
        names; models built from a bare class count carry generated
        ``label_N`` placeholders, for which this returns None so callers
        fall back to their own table.
        """
        names = getattr(getattr(self.model, "label_info", None), "label_names", None)
        if not names or all(_PLACEHOLDER_NAME.match(name) for name in names):
            return None
        return dict(enumerate(names))

    def detect(self, frame_bgr: np.ndarray, frame_id: int) -> Detections:
        """Run one BGR frame through the model.

        Composes `preprocess`, the model's ``predict_step``, and
        `to_detections` into the full frame-to-detections round trip.
        Requires getitune and torch.

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 frame in BGR order.
            frame_id: Frame index to stamp on the returned `Detections`.

        Returns:
            `Detections` for the frame, in original frame coordinates.
        """
        import torch

        with torch.no_grad():
            preds = self.model.predict_step(self.preprocess(frame_bgr), batch_idx=0)
        return self.to_detections(preds, frame_id=frame_id)

    def preprocess(self, frame_bgr: np.ndarray) -> Any:  # noqa: ANN401
        """Preprocess a BGR frame into a getitune ``SampleBatch``.

        Applies the resize and normalization described by the model's
        ``data_input_params`` and sets ``scale_factor`` so predicted
        boxes map back to the original frame coordinates. Requires
        getitune and torch.

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 frame in BGR order, e.g. from
                ``cv2.VideoCapture``.

        Returns:
            A single-image getitune ``SampleBatch`` on ``self.device``.
        """
        import torch
        from getitune.data.entity import ImageInfo, SampleBatch

        inp_h, inp_w = self.model.data_input_params.input_size
        mean = self.model.data_input_params.mean
        std = self.model.data_input_params.std
        ori_h, ori_w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(cv2.resize(rgb, (inp_w, inp_h))).permute(2, 0, 1).float()
        # Mean values below 1.0 indicate the model expects 0-1 normalized pixels
        # (e.g. RF-DETR); otherwise it expects the raw 0-255 range (e.g. YOLOX).
        if all(m < 1.0 for m in mean):
            tensor = tensor / 255.0
        tensor = (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)

        img_info = ImageInfo(
            img_idx=0,
            img_shape=(inp_h, inp_w),
            ori_shape=(ori_h, ori_w),
            scale_factor=(inp_h / ori_h, inp_w / ori_w),
        )
        return SampleBatch(images=tensor.unsqueeze(0).to(self.device), imgs_info=[img_info])

    @staticmethod
    def to_detections(batch: Any, frame_id: int, image_index: int = 0) -> Detections:  # noqa: ANN401
        """Convert one image's predictions from a getitune ``PredictionBatch``.

        Duck-typed: works without getitune or torch installed, on any
        object with per-image ``bboxes``, ``scores``, and ``labels`` lists.

        Args:
            batch: A getitune ``PredictionBatch`` with per-image ``bboxes``,
                ``scores``, and ``labels`` lists (torch tensors or arrays).
            frame_id: Frame index to stamp on the returned `Detections`.
            image_index: Which image of the batch to convert.

        Returns:
            `Detections` with float32 xyxy boxes, float32 scores, and int64
            class ids.

        Raises:
            ValueError: If the batch is missing bboxes, scores, or labels.
        """
        if batch.bboxes is None or batch.scores is None or batch.labels is None:
            msg = "PredictionBatch must carry bboxes, scores, and labels"
            raise ValueError(msg)
        return Detections(
            bboxes=_to_numpy(batch.bboxes[image_index]).reshape(-1, 4).astype(np.float32),
            scores=_to_numpy(batch.scores[image_index]).reshape(-1).astype(np.float32),
            class_ids=_to_numpy(batch.labels[image_index]).reshape(-1).astype(np.int64),
            frame_id=frame_id,
        )


def _to_numpy(value: Any) -> np.ndarray:  # noqa: ANN401
    """Convert a torch tensor (possibly on an accelerator) or array-like to numpy."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
