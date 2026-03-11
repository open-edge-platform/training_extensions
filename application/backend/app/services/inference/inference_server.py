# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from loguru import logger
from model_api.models import Model

from app.models import BatchInferenceInput, BatchInferenceMedia, BatchInferencePrediction, BatchInferenceResult, Label
from app.models.inference import InferenceModel, InferenceState, InferenceStatus
from app.models.model_revision import ModelFormat
from app.services import ModelService, ResourceNotFoundError, ResourceType
from app.services.data_collect.prediction_converter import convert_prediction

MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class _LoadedModel:
    id: UUID
    model: Model
    device: str
    ttl: int
    load_timestamp: datetime


class InferenceServer:
    """
    Inference Server manages the lifecycle of a loaded model for inference, including loading models,
    tracking their status, and performing batch inference.
    It interacts with the ModelService to retrieve model files and uses the Model API to run inference on input data.
    The server maintains the currently loaded model and its associated metadata, such as device and TTL,
    to ensure efficient inference operations.
    """

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service
        self._loaded_model: _LoadedModel | None = None
        self._lock = threading.Lock()

    def set_inference_model(self, project_id: UUID, model_id: UUID, device: str, ttl: int) -> bool:
        with self._lock:
            if self._loaded_model and self._loaded_model.id == model_id and self._loaded_model.device == device:
                return False  # same model and device, no need to reload

            format = ModelFormat.OPENVINO
            logger.info("Loading model {} with format {} on device {}", model_id, format, device)
            files_exist, paths = self._model_service.get_model_binary_files(
                project_id=project_id, model_id=model_id, format=format
            )
            if not files_exist:
                raise ResourceNotFoundError(
                    resource_type=ResourceType.MODEL,
                    resource_id=f"{model_id} with format {format.value}",
                )
            model_xml_path, _ = paths

            model = Model.create_model(
                model=str(model_xml_path),
                device=device,
                nstreams=MODELAPI_NSTREAMS,
            )

            self._loaded_model = _LoadedModel(
                id=model_id, model=model, device=device, ttl=ttl, load_timestamp=datetime.now()
            )
            return True

    def get_status(self) -> InferenceState:
        with self._lock:
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
        with self._lock:
            if self._loaded_model is None:
                raise RuntimeError("No model loaded for inference")
            logger.debug("Running inference on batch of {} inputs", len(inputs))

            input_data = [input.data for input in inputs]
            inference_result = self._loaded_model.model.infer_batch(input_data)
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
        with self._lock:
            self._loaded_model = None
