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
from app.models.model_revision import ModelFormat, ModelPrecision
from app.services import ModelService, ResourceNotFoundError, ResourceType
from app.services.data_collect.prediction_converter import convert_prediction

MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class _LoadedModel:
    id: UUID
    model: Model
    device: str
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
        self._loading_model: bool = False
        self._lock = threading.Lock()

    def set_inference_model(self, project_id: UUID, model_id: UUID, device: str, ttl: int) -> bool:
        self._lock.acquire(timeout=10)
        try:
            if self._loaded_model and self._loaded_model.id == model_id and self._loaded_model.device == device:
                return False  # same model and device, no need to reload

            format = ModelFormat.OPENVINO
            precision = ModelPrecision.FP16
            logger.info("Loading model {} with format {} on device {}", model_id, format, device)
            model_variants = self._model_service.get_model_variants(project_id=project_id, model_id=model_id)
            model_variant = next(
                (mv for mv in model_variants if mv.format == format and mv.precision == precision), None
            )
            if not model_variant:
                raise ResourceNotFoundError(
                    resource_type=ResourceType.MODEL,
                    resource_id=f"{model_id} with format {format.value} and precision {precision}",
                )
            files_exist, paths = self._model_service.get_model_binary_files(
                project_id=project_id, model_id=model_id, model_variant_id=model_variant.id
            )
            if not files_exist:
                raise ResourceNotFoundError(
                    resource_type=ResourceType.MODEL,
                    resource_id=f"{model_id} with format {format.value}",
                )
            model_xml_path, _ = paths

            self._loading_model = True
            model = Model.create_model(
                model=str(model_xml_path),
                device=device,
                nstreams=MODELAPI_NSTREAMS,
            )
            self._loading_model = False

            self._loaded_model = _LoadedModel(id=model_id, model=model, device=device, load_timestamp=datetime.now())
            return True
        finally:
            self._lock.release()

    def get_status(self) -> InferenceState:
        """
        Get inference server status

        Returns:
            Inference server status
        """
        loaded_model = self._loaded_model
        loading_model = self._loading_model
        if loaded_model is None:
            return InferenceState(status=(InferenceStatus.IDLE if not loading_model else InferenceStatus.LOADING))

        return InferenceState(
            status=InferenceStatus.ACTIVE,
            model=InferenceModel(
                model_id=loaded_model.id,
                device=loaded_model.device,
                load_timestamp=loaded_model.load_timestamp,
            ),
        )

    def infer_batch(self, labels: list[Label], inputs: list[BatchInferenceInput]) -> BatchInferenceResult:
        """
        Perform batch inference on the provided inputs using the currently loaded model.
        It processes each input, runs inference, and converts the raw predictions into a structured
        format using the provided labels.

        Args:
            labels: Project labels
            inputs: List of inputs

        Returns:
            Prediction results for inputs
        """
        if self._loaded_model is None:
            raise RuntimeError("No model loaded for inference")
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
        """
        Stop inference server and unload the model.
        """
        # TODO: model unload & active inference cancellation
        with self._lock:
            self._loaded_model = None
