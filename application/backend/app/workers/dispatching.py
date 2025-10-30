# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass

from app.db import get_db_session
from app.schemas import DisconnectedSinkConfig, Sink, SinkType
from app.services import DispatchService
from app.services.configuration_service import SinkService
from app.services.data_collect import DataCollector
from app.services.dispatchers import Dispatcher
from app.services.event.event_bus import EventBus, EventType
from app.stream.stream_data import StreamData
from app.workers.base import BaseThreadWorker

logger = logging.getLogger(__name__)


class DispatchingWorker(BaseThreadWorker):
    """
    A thread that pulls predictions from the queue and dispatches them to the configured outputs
    and WebRTC visualization stream.
    """

    ROLE = "Dispatching"

    def __init__(
        self,
        event_bus: EventBus,
        pred_queue: mp.Queue,
        rtc_stream_queue: queue.Queue,
        stop_event: EventClass,
        data_collector: DataCollector,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._pred_queue = pred_queue
        self._rtc_stream_queue = rtc_stream_queue

        self._data_collector = data_collector

        self._sink: Sink
        self._destinations: list[Dispatcher] = []

        self._sink, self._destinations = self._load_sink()
        logger.info(f"Active sink set to {self._sink}")
        event_bus.subscribe([EventType.SINK_CHANGED], self._reload_sink)

    def setup(self) -> None:
        pass

    def _load_sink(self) -> tuple[Sink, list[Dispatcher]]:
        with get_db_session() as db:
            active_sink = SinkService(db).get_active_sink()
        sink = active_sink if active_sink is not None else DisconnectedSinkConfig()
        destinations = DispatchService.get_destinations(output_configs=[sink])
        return sink, destinations

    def _reload_sink(self) -> None:
        self._sink, self._destinations = self._load_sink()
        logger.info(f"Active sink set to {self._sink}")

    def run_loop(self) -> None:
        while not self.should_stop():
            if self._sink.sink_type == SinkType.DISCONNECTED:
                logger.debug("No sink available... retrying in 1 second")
                self.stop_aware_sleep(1)
                continue

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

            # Collect the image to project dataset if needed
            self._data_collector.collect(
                timestamp=stream_data.timestamp,
                frame_data=stream_data.frame_data,
                inference_data=inference_data,
            )
