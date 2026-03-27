# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from uuid import uuid4

import numpy as np
from datumaro.experimental import LazyImage, LazyVideoFrame, MediaInfo
from datumaro.experimental.fields import Subset

from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.models import Task, TaskType


class SampleFactory:
    """Knows how to build an annotated sample for a given task type."""

    def __init__(self, task: Task) -> None:
        self._task = task

    def build(
        self,
        label_idx: int,
        subset: str,
        lazy_media: LazyImage | LazyVideoFrame,
        media_info: MediaInfo,
    ):
        """Return a fully constructed sample for the task's annotation type."""
        base = {
            "id": str(uuid4()),
            "media": lazy_media,
            "media_info": media_info,
            "subset": Subset[subset.upper()],
            "user_reviewed": True,
        }
        match self._task.task_type:
            case TaskType.DETECTION:
                return DetectionImportExportSample(
                    **base,
                    confidence=None,
                    label=np.array([label_idx]),
                    bboxes=np.array([[*self._random_bbox()]]),
                )
            case TaskType.CLASSIFICATION:
                if self._task.exclusive_labels:
                    return MulticlassClassificationImportExportSample(**base, confidence=None, label=label_idx)
                return MultilabelClassificationImportExportSample(**base, confidence=None, label=np.array([label_idx]))
            case TaskType.INSTANCE_SEGMENTATION:
                return InstanceSegmentationImportExportSample(
                    **base,
                    confidence=None,
                    label=np.array([label_idx]),
                    polygons=np.array([self._random_polygon()]),
                )

    @staticmethod
    def _random_bbox() -> tuple[int, int, int, int]:
        x1, y1 = 10 + secrets.randbelow(50), 20 + secrets.randbelow(50)
        return x1, y1, x1 + 80 + secrets.randbelow(100), y1 + 100 + secrets.randbelow(150)

    @staticmethod
    def _random_polygon() -> list[list[int]]:
        return [
            [10 + secrets.randbelow(50), 20 + secrets.randbelow(50)],
            [60 + secrets.randbelow(50), 20 + secrets.randbelow(50)],
            [60 + secrets.randbelow(50), 120 + secrets.randbelow(150)],
            [10 + secrets.randbelow(50), 120 + secrets.randbelow(150)],
        ]
