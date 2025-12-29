# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import Any

import cv2
from loguru import logger
from loguru._logger import Logger as LoguruLogger
from model_api.models import Model
from model_api.models.result import Result

from app.services import ActiveModelService, MetricsService
from app.services.active_model_service import LoadedModel
from app.settings import Settings, get_settings
from app.stream.stream_data import InferenceData, StreamData
from app.utils import Visualizer
from app.workers.base import BaseProcessWorker


@dataclass(frozen=True, kw_only=True)
class InferenceWorkerConfig:
    frame_queue: mp.Queue
    pred_queue: mp.Queue
    stop_event: EventClass
    model_reload_event: EventClass | None
    shm_name: str
    shm_lock: Lock
    logger_: LoguruLogger


class InferenceWorker(BaseProcessWorker):
    """A process that pulls frames from the frame queue, runs inference, and pushes results to the prediction queue."""

    ROLE = "Inference"

    def __init__(
        self,
        config: InferenceWorkerConfig,
    ) -> None:
        super().__init__(stop_event=config.stop_event, logger_=config.logger_, queues_to_cancel=[config.pred_queue])
        self._frame_queue = config.frame_queue
        self._pred_queue = config.pred_queue
        self._model_reload_event = config.model_reload_event
        self._shm_name = config.shm_name
        self._shm_lock = config.shm_lock

        self._settings: Settings | None = None
        self._metrics_service: MetricsService | None = None
        self._model_service: ActiveModelService | None = None
        self._loaded_model: LoadedModel | None = None
        self._last_model_obj_id = 0  # track the id of the Model object to install the callback only once

    def setup(self) -> None:
        super().setup()
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)
        self._model_service = ActiveModelService(get_settings().data_dir)

    def _on_inference_completed(self, inf_result: Result, userdata: dict[str, Any]) -> None:
        start_time = float(userdata["inference_start_time"])
        model_id = userdata["model_id"]
        self._metrics_service.record_inference_end(model_id=model_id, start_time=start_time)  # type: ignore

        stream_data: StreamData = userdata["stream_data"]
        frame_with_predictions = Visualizer.overlay_predictions(
            original_image=stream_data.frame_data, predictions=inf_result
        )
        inference_data = InferenceData(
            prediction=inf_result,
            visualized_prediction=frame_with_predictions,
            model_id=model_id,
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
        logger.debug("Installed inference callback for model object with id '{}'", self._last_model_obj_id)

    def _refresh_loaded_model(self) -> LoadedModel | None:
        """
        Get (or reload) the active model. If reloads are requested repeatedly,
        clear the event until the latest model is loaded.
        """
        # If no reload requested, return current model
        if self._model_reload_event is None or not self._model_reload_event.is_set():
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
                        cv2.cvtColor(item.frame_data, cv2.COLOR_BGR2RGB),  # models expect RGB input
                        user_data={
                            "stream_data": item,
                            "model_id": self._loaded_model.id,
                            "inference_start_time": inference_start_time,
                        },
                    )
                else:
                    model.inference_adapter.await_any()
            except Exception:
                logger.exception("Unhandled error in inference loop")
                self.stop_aware_sleep(2)
