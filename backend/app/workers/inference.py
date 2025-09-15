# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import Any
from uuid import UUID

from model_api.models import DetectionResult, Model

from app.entities.stream_data import InferenceData, StreamData
from app.services import ModelService
from app.services.metrics_service import MetricsService
from app.services.model_service import LoadedModel
from app.utils import Visualizer
from app.workers.base import BaseProcessWorker

logger = logging.getLogger(__name__)


class InferenceWorker(BaseProcessWorker):
    """A process that pulls frames from the frame queue, runs inference, and pushes results to the prediction queue."""

    ROLE = "Inference"

    def __init__(
        self,
        frame_queue: mp.Queue,
        pred_queue: mp.Queue,
        stop_event: EventClass,
        model_reload_event: EventClass,
        shm_name: str,
        shm_lock: Lock,
    ) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[pred_queue])
        self._frame_queue = frame_queue
        self._pred_queue = pred_queue
        self._model_reload_event = model_reload_event
        self._shm_name = shm_name
        self._shm_lock = shm_lock

        self._metrics_service: MetricsService | None = None
        self._model_service: ModelService | None = None
        self._loaded_model: LoadedModel | None = None
        self._last_model_obj_id = 0  # track the id of the Model object to install the callback only once

    def setup(self) -> None:
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)
        self._model_service = ModelService()

    def _on_inference_completed(self, inf_result: DetectionResult, userdata: dict[str, Any]) -> None:
        start_time = float(userdata["inference_start_time"])
        model_id = UUID(userdata["model_id"])
        self._metrics_service.record_inference_end(model_id=model_id, start_time=start_time)  # type: ignore

        stream_data: StreamData = userdata["stream_data"]
        frame_with_predictions = Visualizer.overlay_predictions(
            original_image=stream_data.frame_data, predictions=inf_result
        )
        inference_data = InferenceData(
            prediction=inf_result,
            visualized_prediction=frame_with_predictions,
            model_name=userdata["model_name"],
        )
        stream_data.inference_data = inference_data
        while not self.should_stop():
            try:
                self._pred_queue.put(stream_data, timeout=1)
                break
            except queue.Full:
                logger.debug("Prediction queue is full, retrying...")

    def _install_callback_if_needed(self, model: Model) -> None:
        """Install inference completion callback once per model object instance."""
        obj_id = id(model)
        if obj_id == self._last_model_obj_id:
            return

        model.set_callback(self._on_inference_completed)
        self._last_model_obj_id = obj_id
        logger.debug("Installed inference callback for model object with id '%s'", self._last_model_obj_id)

    def _refresh_loaded_model(self) -> LoadedModel | None:
        """
        Get (or reload) the active model. If reloads are requested repeatedly,
        clear the event until the latest model is loaded.
        """
        # If no reload requested, return current model
        if not self._model_reload_event.is_set():
            return self._model_service.get_loaded_inference_model()  # type: ignore

        # Process reload requests - keep reloading until event stabilizes
        loaded_model = None
        while self._model_reload_event.is_set():
            self._model_reload_event.clear()
            loaded_model = self._model_service.get_loaded_inference_model(force_reload=True)  # type: ignore
        return loaded_model

    def run_loop(self) -> None:
        while not self.should_stop():
            try:
                self._loaded_model = self._refresh_loaded_model()
                if self._loaded_model is None:
                    logger.debug("No model available... retrying in 1 second")
                    self.stop_aware_sleep(1)
                    continue

                model = self._loaded_model.model
                self._install_callback_if_needed(model)
                if model.inference_adapter.is_ready():
                    try:
                        item = self._frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    inference_start_time = self._metrics_service.record_inference_start()  # type: ignore
                    model.infer_async(
                        item.frame_data,
                        user_data={
                            "stream_data": item,
                            "model_name": self._model_service.get_active_model_name(),  # type: ignore
                            "model_id": str(self._loaded_model.id),
                            "inference_start_time": inference_start_time,
                        },
                    )
                else:
                    model.inference_adapter.await_any()
            except Exception:
                logger.exception("Unhandled error in inference loop")
                self.stop_aware_sleep(2)
