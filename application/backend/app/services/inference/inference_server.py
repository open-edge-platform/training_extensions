# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import threading
from pathlib import Path
from uuid import UUID

from loguru import logger

from app.db import get_db_session
from app.models import BatchInferenceInput, DatasetItemAnnotation, Label
from app.models.inference import InferenceModel, InferenceState, InferenceStatus
from app.models.model_revision import ModelFormat, ModelPrecision
from app.models.system import DeviceInfo
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.data_collect.prediction_converter import convert_prediction

from .model_loader import LoadedModelHandle, ModelLoader


class InferenceBusyError(Exception):
    """
    Exception raised when an inference request is made while the server is busy loading a model or performing inference.
    """

    def __init__(self):
        super().__init__(
            "Inference request timed out waiting for the model lock. Another inference is in "
            "progress or model is not loaded yet."
        )


class InferenceServer:
    """
    Inference Server manages the lifecycle of a loaded model for inference, including loading models,
    tracking their status, and performing batch inference.
    It interacts with the ModelService to retrieve model files and uses the Model API to run inference on input data.
    The server maintains the currently loaded model and its associated metadata, such as device,
    to ensure efficient inference operations.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._loaded_model: LoadedModelHandle | None = None
        self._loading_model: bool = False
        self._lock = threading.Lock()

    def set_inference_model(self, project_id: UUID, model_id: UUID, device: DeviceInfo, ttl: int) -> bool:
        """
        Load the specified model for inference.
        If the same model is already loaded on the same device, it does nothing.

        Args:
            project_id: Project identifier.
            model_id: Model identifier.
            device: Device to use for inference.
            ttl: Model time-to-live (TTL).

        Returns:
            True, if the model has been loaded, False if model has not been changed.
        """
        self._lock.acquire()
        try:
            if self._loaded_model and self._loaded_model.model_id == model_id and self._loaded_model.device == device:
                return False  # same model and device, no need to reload
            self._loading_model = True

            with get_db_session() as db:
                from app.services import ModelService

                model_service = ModelService(data_dir=self._data_dir, db_session=db)

                format = ModelFormat.OPENVINO
                precision = ModelPrecision.FP16
                logger.info("Loading model {} with format {} on device {}", model_id, format, device)
                model_variants = model_service.get_model_variants(project_id=project_id, model_id=model_id)
                model_variant = next(
                    (mv for mv in model_variants if mv.format == format and mv.precision == precision), None
                )
                if not model_variant:
                    raise ResourceNotFoundError(
                        resource_type=ResourceType.MODEL,
                        resource_id=f"{model_id} with format {format.value} and precision {precision}",
                    )
                files_exist, paths = model_service.get_model_binary_files(
                    project_id=project_id, model_id=model_id, model_variant_id=model_variant.id
                )
                if not files_exist:
                    raise ResourceNotFoundError(
                        resource_type=ResourceType.MODEL,
                        resource_id=f"{model_id} with format {format.value}",
                    )
                model_xml_path, _ = paths

                if self._loaded_model is not None:
                    ModelLoader.unload(self._loaded_model)
                    self._loaded_model = None
                self._loaded_model = ModelLoader.load(
                    model_id=model_id, variant_id=model_variant.id, model_xml_path=model_xml_path, device=device
                )
                return True
        finally:
            self._loading_model = False
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
                model_id=loaded_model.model_id,
                device=loaded_model.device,
                load_timestamp=loaded_model.loaded_at,
            ),
        )

    def infer_batch(
        self, labels: list[Label], inputs: list[BatchInferenceInput]
    ) -> dict[tuple[UUID, int | None], list[DatasetItemAnnotation]]:
        """
        Perform batch inference on the provided inputs using the currently loaded model.
        It processes each input, runs inference, and converts the raw predictions into a structured
        format using the provided labels.

        Args:
            labels: Project labels
            inputs: List of inputs

        Returns:
            Dictionary mapping (media_id, frame_index) tuples to lists of DatasetItemAnnotation predictions.
        """
        if self._loaded_model is None:
            raise RuntimeError("No model loaded for inference")
        if not self._lock.acquire(timeout=30):
            raise InferenceBusyError
        try:
            if self._loaded_model is None:
                raise RuntimeError("No model loaded for inference")
            logger.debug("Running inference on batch of {} inputs", len(inputs))

            input_data = [inp.data for inp in inputs]
            inference_result = self._loaded_model.model.infer_batch(input_data)
        finally:
            self._lock.release()

        return {
            (input.media_id, input.frame_index): convert_prediction(
                labels=labels, frame_data=input_data[idx], prediction=inference_result[idx]
            )
            for idx, input in enumerate(inputs)
        }

    def stop(self) -> None:
        """
        Stop inference server and unload the model.
        """
        # TODO: model unload & active inference cancellation
        with self._lock:
            if self._loaded_model is not None:
                ModelLoader.unload(self._loaded_model)
                self._loaded_model = None
