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
    Information about model used for inference it's load time and device.
    """

    model_id: UUID
    device: str
    load_timestamp: datetime


class InferenceStatus(StrEnum):
    """
    Inference server status. Can be idle, loading or active
    """

    IDLE = "IDLE"
    LOADING = "LOADING"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class InferenceState:
    """
    Inference server status.
    Contains information about the current inference state and model if loaded.
    """

    status: InferenceStatus
    model: InferenceModel | None = None
