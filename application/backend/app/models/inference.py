# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
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


@dataclass(frozen=True)
class InferenceModel:
    """
    Information about model used for inference with it's TTL, device and remaining time before unload.
    """

    model_id: UUID
    device: str
    ttl: int
    load_timestamp: datetime
    remaining_seconds: float


class InferenceStatus(StrEnum):
    """
    Inference server status. Either idle or active
    """

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class InferenceState:
    """
    Inference server status.
    Contains information about the current inference state and model if loaded.
    """

    status: InferenceStatus
    model: InferenceModel | None = None
