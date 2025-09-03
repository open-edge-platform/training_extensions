# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass
from typing import Any
from uuid import UUID

from model_api.models import DetectionResult

from app.entities.stream_data import InferenceData, StreamData
from app.services import ModelService
from app.services.metrics_service import MetricsService
from app.services.model_service import LoadedModel
from app.utils import Visualizer, flush_queue, log_threads, suppress_child_shutdown_signals

logger = logging.getLogger(__name__)


def inference_routine(  # noqa: C901, PLR0915
    frame_queue: mp.Queue, pred_queue: mp.Queue, stop_event: EventClass, model_reload_event: EventClass
) -> None:
    """Load frames from the frame queue, run inference then inject the result into the predictions queue"""
    suppress_child_shutdown_signals()

    metrics_collector = MetricsService()

    def on_inference_completed(inf_result: DetectionResult, userdata: dict[str, Any]) -> None:
        start_time = float(userdata["inference_start_time"])
        model_id = UUID(userdata["model_id"])
        metrics_collector.record_inference_end(model_id=model_id, start_time=start_time)

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
        while not stop_event.is_set():
            try:
                pred_queue.put(stream_data, timeout=1)
                break
            except queue.Full:
                logger.debug("Prediction queue is full, retrying...")

    model_service = ModelService()
    loaded_model: LoadedModel | None = None
    last_model_id: int = 0  # track the id of the Model object to install the callback only once

    try:
        while not stop_event.is_set():
            # Get the model, reloading it if necessary
            if not model_reload_event.is_set():
                loaded_model = model_service.get_loaded_inference_model()
            else:
                # The 'while' loop handles the case when the active model is switched again while reloading.
                while model_reload_event.is_set():
                    model_reload_event.clear()
                    loaded_model = model_service.get_loaded_inference_model(force_reload=True)

            if loaded_model is None:
                logger.debug("No model available... retrying in 1 second")
                time.sleep(1)
                continue

            model = loaded_model.model
            # Install the callback if it's the first iteration with this model
            if id(model) != last_model_id:
                model.set_callback(on_inference_completed)
                last_model_id = id(model)
                logger.debug(f"Installed inference callback for model object with id '{last_model_id}'")

            if model.inference_adapter.is_ready():
                try:
                    queue_data = frame_queue.get(timeout=1)
                except queue.Empty:
                    continue

                inference_start_time = metrics_collector.record_inference_start()
                model.infer_async(
                    queue_data.frame_data,
                    user_data={
                        "stream_data": queue_data,
                        "model_name": model_service.get_active_model_name(),
                        "model_id": str(loaded_model.id),
                        "inference_start_time": inference_start_time,
                    },
                )
            else:
                model.inference_adapter.await_any()
    finally:
        # Empty the prediction queue to ensure the termination of QueueFeederThread (internal thread of 'mp.Queue')
        if pred_queue is not None:
            logger.debug("Flushing the pred queue from leftover frames")
            flush_queue(pred_queue)
            del pred_queue

        log_threads(log_level=logging.DEBUG)
        logger.info("Stopped inference routine")
