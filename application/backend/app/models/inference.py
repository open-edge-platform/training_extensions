# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from app.models import DatasetItemAnnotation


@dataclass(frozen=True)
class BatchInferenceInput:
    media_id: UUID
    data: np.ndarray
    frame_index: int | None = None


@dataclass(frozen=True)
class BatchInferenceMedia:
    id: UUID
    frame_index: int | None = None


@dataclass(frozen=True)
class BatchInferencePrediction:
    media: BatchInferenceMedia
    prediction: list[DatasetItemAnnotation]


@dataclass(frozen=True)
class BatchInferenceResult:
    predictions: list[BatchInferencePrediction]
