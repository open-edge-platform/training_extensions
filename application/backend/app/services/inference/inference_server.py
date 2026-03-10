# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from loguru import logger
from model_api.models import Model

from app.models import BatchInferenceInput, BatchInferenceMedia, BatchInferencePrediction, BatchInferenceResult, Label
from app.services.data_collect.prediction_converter import convert_prediction

MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class LoadedModel:
    id: UUID
    model: Model
    device: str
    ttl: int
    load_timestamp: datetime


@dataclass(frozen=True)
class InferenceModel:
    model_id: UUID
    device: str
    ttl: int
    load_timestamp: datetime
    remaining_seconds: float


class InferenceStatus(StrEnum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class InferenceState:
    status: InferenceStatus
    model: InferenceModel | None = None


class InferenceServer:
    def __init__(self, data_dir: Path) -> None:
        self._projects_dir = data_dir / "projects"
        self._loaded_model: LoadedModel | None = None

    def _get_model_file_path(self, project_id: UUID, model_id: UUID, extension: str = "xml") -> Path:
        file_path = self._projects_dir / f"{project_id}/models/{model_id}/model.{extension}"
        if not file_path.is_file():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        return file_path

    def set_inference_model(self, project_id: UUID, model_id: UUID, device: str, ttl: int) -> bool:
        if self._loaded_model and self._loaded_model.id == model_id and self._loaded_model.device == device:
            return False  # same model and device, no need to reload

        logger.info("Loading model {} on device {}", model_id, device)
        model_xml_path = self._get_model_file_path(project_id, model_id, "xml")
        _ = self._get_model_file_path(project_id, model_id, "bin")

        model = Model.create_model(
            model=str(model_xml_path),
            device=device,
            nstreams=MODELAPI_NSTREAMS,
        )

        self._loaded_model = LoadedModel(
            id=model_id, model=model, device=device, ttl=ttl, load_timestamp=datetime.now()
        )
        return True

    def get_status(self) -> InferenceState:
        if self._loaded_model is None:
            return InferenceState(status=InferenceStatus.IDLE)

        return InferenceState(
            status=InferenceStatus.ACTIVE,
            model=InferenceModel(
                model_id=self._loaded_model.id,
                device=self._loaded_model.device,
                ttl=self._loaded_model.ttl,
                load_timestamp=self._loaded_model.load_timestamp,
                remaining_seconds=self._loaded_model.ttl
                - (datetime.now() - self._loaded_model.load_timestamp).total_seconds(),
            ),
        )

    def infer_batch(self, labels: list[Label], inputs: list[BatchInferenceInput]) -> BatchInferenceResult:
        loaded_model = self._loaded_model
        if loaded_model is None:
            raise RuntimeError("No model loaded for inference")
        logger.debug("Running inference on batch of {} inputs", len(inputs))

        input_data = [input.data for input in inputs]
        inference_result = loaded_model.model.infer_batch(input_data)
        result = BatchInferenceResult(predictions=[])
        for idx, input in enumerate(inputs):
            result.predictions.append(
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=input.media_id, frame_index=input.frame_index),
                    prediction=convert_prediction(
                        labels=labels, frame_data=input_data[idx], prediction=inference_result[idx]
                    ),
                )
            )
        return result

    def stop(self) -> None:
        # TODO: model unload & active inference cancellation
        self._loaded_model = None
