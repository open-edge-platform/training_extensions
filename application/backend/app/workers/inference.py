# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Any

import cv2
from loguru import logger
from loguru._logger import Logger as LoguruLogger

from app.services import ActiveModelService, MetricsService
from app.services.inference.model_loader import LoadedModelHandle
from app.settings import Settings, get_settings
from app.stream.stream_data import InferenceData, StreamData
from app.utils import Visualizer
from app.workers.base import BaseProcessWorker

if TYPE_CHECKING:
    from model_api.models import Model
    from model_api.models.result import Result


@dataclass(frozen=True, kw_only=True)
class InferenceWorkerConfig:
    frame_queue: mp.Queue
    pred_queue: mp.Queue
    stop_event: EventClass
    model_reload_event: EventClass | None
    shm_name: str
    shm_lock: Lock
    logger_: LoguruLogger


class PredictionReorderBuffer:
    """Buffers predictions to ensure async generated predictions are fed to the prediction queue in order"""

    def __init__(self, max_size: int = 10):
        self._buffer: dict[float, StreamData] = {}
        self._expected_timestamps: list[float] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def register_expected_timestamp(self, timestamp: float) -> None:
        with self._lock:
            self._expected_timestamps.append(timestamp)
            if len(self._expected_timestamps) > self._max_size:
                oldest_timestamp = self._expected_timestamps.pop(0)
                if oldest_timestamp in self._buffer:
                    del self._buffer[oldest_timestamp]

    def add_prediction_for_timestamp(self, timestamp: float, stream_data: StreamData) -> None:
        with self._lock:
            if timestamp in self._expected_timestamps:
                self._buffer[timestamp] = stream_data
            else:
                logger.warning(
                    "Received prediction for buffering with unexpected timestamp: "
                    "expected timestamps: {}, received timestamp {}",
                    self._expected_timestamps,
                    timestamp,
                )

    def get_ready_predictions(self) -> list[StreamData]:
        result = []
        with self._lock:
            while self._expected_timestamps and self._expected_timestamps[0] in self._buffer:
                timestamp = self._expected_timestamps.pop(0)
                stream_data = self._buffer.pop(timestamp)
                result.append(stream_data)
        return result

    def clear(self) -> None:
        with self._lock:
            self._expected_timestamps = []
            self._buffer = {}


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
        self._loaded_model: LoadedModelHandle | None = None
        self._last_model_obj_id = 0  # track the id of the Model object to install the callback only once

        # The reorder buffer ensures that frames are broadcast in order, even if inference results are produced
        # out of order due to async processing. It is initialized in setup() to ensure it's created after the process
        # fork, since it contains a threading.Lock that can't be pickled.
        self.__prediction_buffer: PredictionReorderBuffer | None = None

    def setup(self) -> None:
        super().setup()
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)
        self._model_service = ActiveModelService(get_settings().data_dir)
        self.__prediction_buffer = PredictionReorderBuffer()

    @property
    def _prediction_buffer(self) -> PredictionReorderBuffer:
        if self.__prediction_buffer is None:
            raise RuntimeError("Prediction buffer not initialized (method 'setup' not called?)")
        return self.__prediction_buffer

    def _on_inference_completed(self, inf_result: "Result", userdata: dict[str, Any]) -> None:
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

        # Predictions are generated async, first add to buffer,
        # then check if next expected predictions are ready to be queued
        self._prediction_buffer.add_prediction_for_timestamp(timestamp=start_time, stream_data=stream_data)
        for stream_data in self._prediction_buffer.get_ready_predictions():
            self._enqueue_prediction(stream_data)

    def _enqueue_prediction(self, stream_data: StreamData) -> None:
        """Push a prediction to the prediction queue without ever blocking.

        This callback runs on the Model API / OpenVINO inference thread. It MUST return
        promptly: ``model_api`` (via OpenVINO's ``AsyncInferQueue``) waits for all in-flight
        requests to complete when the model is unloaded/reloaded. If this method blocked on a
        full prediction queue (e.g. because the downstream dispatcher is momentarily stalled on
        a DB or disk write), the model unload would deadlock and the whole inference process
        would wedge - never loading another model again.

        Since the visualization stream is live, we use drop-oldest semantics: when the queue is
        full we evict the stalest prediction in favour of the newest one.
        """
        try:
            self._pred_queue.put_nowait(stream_data)
            return
        except queue.Full:
            pass

        # Queue full: drop the oldest prediction to make room for the newest one.
        try:
            self._pred_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._pred_queue.put_nowait(stream_data)
        except queue.Full:
            logger.debug("Prediction queue still full, dropping prediction")

    def _install_callback_if_needed(self, model: "Model") -> None:
        """Install inference completion callback once per model object instance."""
        obj_id = id(model)
        if obj_id == self._last_model_obj_id:
            return

        model.set_callback(self._on_inference_completed)
        self._last_model_obj_id = obj_id
        logger.debug("Installed inference callback for model object with id '{}'", self._last_model_obj_id)

    def _refresh_loaded_model(self) -> LoadedModelHandle | None:
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

        # Always clear the prediction buffer on reload to prevent stale timestamps
        # from blocking future predictions (e.g. in-flight async inferences that were
        # cancelled when the old model was unloaded will never produce callbacks).
        self._prediction_buffer.clear()

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
                    self._prediction_buffer.register_expected_timestamp(inference_start_time)
                    rgb_frame = cv2.cvtColor(item.frame_data, cv2.COLOR_BGR2RGB)
                    model.infer_async(
                        rgb_frame,
                        user_data={
                            "stream_data": item,
                            "model_id": self._loaded_model.model_id,
                            "inference_start_time": inference_start_time,
                        },
                    )
                else:
                    model.inference_adapter.await_any()
            except Exception:
                logger.exception("Unhandled error in inference loop")
                self.stop_aware_sleep(2)
