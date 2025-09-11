# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass

from app.entities.stream_data import StreamData
from app.schemas import Sink, SinkType
from app.services import ActivePipelineService, DispatchService
from app.services.dispatchers import Dispatcher
from app.workers.base import BaseThreadWorker

logger = logging.getLogger(__name__)


class DispatchingWorker(BaseThreadWorker):
    """
    A thread that pulls predictions from the queue and dispatches them to the configured outputs
    and WebRTC visualization stream.
    """

    ROLE = "Dispatching"

    def __init__(self, pred_queue: mp.Queue, rtc_stream_queue: queue.Queue, stop_event: EventClass) -> None:
        super().__init__(stop_event=stop_event)
        self._pred_queue = pred_queue
        self._rtc_stream_queue = rtc_stream_queue

        self._active_pipeline_service: ActivePipelineService | None = None
        self._prev_sink_config: Sink | None = None
        self._destinations: list[Dispatcher] = []

    def setup(self) -> None:
        from app.api.dependencies import get_active_pipeline_service  # Avoid circular import

        self._active_pipeline_service = get_active_pipeline_service()

    def _reset_sink_if_needed(self, sink_config: Sink) -> None:
        if not self._prev_sink_config or sink_config != self._prev_sink_config:
            logger.debug("Sink config changed from %s to %s", self._prev_sink_config, sink_config)
            self._destinations = DispatchService.get_destinations(output_configs=[sink_config])
            self._prev_sink_config = copy.deepcopy(sink_config)

    def run_loop(self) -> None:
        while not self.should_stop():
            sink_config = self._active_pipeline_service.get_sink_config()  # type: ignore

            if sink_config.sink_type == SinkType.DISCONNECTED:
                logger.debug("No sink available... retrying in 1 second")
                self.stop_aware_sleep(1)
                continue

            self._reset_sink_if_needed(sink_config)

            # Read from the queue
            try:
                stream_data: StreamData = self._pred_queue.get(timeout=1)
            except queue.Empty:
                logger.debug("Nothing to dispatch yet")
                continue

            if stream_data.inference_data is None:
                logger.error("Missing inference data in stream_data; skipping dispatch")
                continue

            inference_data = stream_data.inference_data
            if inference_data is None:
                logger.error("No inference data available")
                continue

            image_with_visualization = inference_data.visualized_prediction
            prediction = inference_data.prediction
            # Postprocess and dispatch results
            for destination in self._destinations:
                destination.dispatch(
                    original_image=stream_data.frame_data,
                    image_with_visualization=image_with_visualization,
                    predictions=prediction,
                )

            # Dispatch to WebRTC stream
            try:
                self._rtc_stream_queue.put(image_with_visualization, block=False)
            except queue.Full:
                logger.debug("Visualization queue is full; skipping")
