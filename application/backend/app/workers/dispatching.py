# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass

import numpy as np
from loguru import logger

from app.db import get_db_session
from app.models import DisconnectedSinkConfig, Sink, SinkType
from app.services import DispatchService, SinkService
from app.services.data_collect import DataCollector
from app.services.dispatchers import Dispatcher
from app.services.event.event_bus import EventBus, EventType
from app.stream.stream_data import StreamData
from app.webrtc import FrameBroadcaster
from app.workers.base import BaseThreadWorker


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
        rtc_stream_broadcaster: FrameBroadcaster[np.ndarray],
        stop_event: EventClass,
        data_collector: DataCollector,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._event_bus = event_bus
        self._pred_queue = pred_queue
        self._rtc_stream_broadcaster = rtc_stream_broadcaster

        self._data_collector = data_collector

        self._sink: Sink
        self._destinations: list[Dispatcher] = []

        self._sink, self._destinations = self._load_sink()
        logger.info(f"Active sink set to {self._sink}")
        event_bus.subscribe(
            [EventType.SINK_CHANGED, EventType.PIPELINE_STATUS_CHANGED],
            self._reload_sink,
        )

    def setup(self) -> None:
        pass

    def _load_sink(self) -> tuple[Sink, list[Dispatcher]]:
        with get_db_session() as db:
            active_sink = SinkService(event_bus=self._event_bus, db_session=db).get_active_sink()
        sink = active_sink if active_sink is not None else DisconnectedSinkConfig()
        destinations = DispatchService.get_destinations(output_configs=[sink])
        return sink, destinations

    def _reload_sink(self) -> None:
        self._sink, self._destinations = self._load_sink()
        logger.info(f"Active sink set to {self._sink}")
        # Drain stale frames from WebRTC consumer queues so that clients immediately
        # receive fresh predictions after a pipeline or sink change instead of replaying
        # frames from the previous configuration.
        self._rtc_stream_broadcaster.clear()

    def run_loop(self) -> None:
        while not self.should_stop():
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

            # Dispatch to the WebRTC stream first so the live preview stays responsive even if
            # an external sink or data collection step below is momentarily slow (e.g. a blocking
            # disk or DB write). broadcast() is non-blocking (drop-oldest per consumer).
            self._rtc_stream_broadcaster.broadcast(image_with_visualization)

            # Postprocess and dispatch results to external sinks (folder, MQTT, ROS, webhook, ...).
            # Skipped when no sink is configured; WebRTC and data collection still run regardless.
            if self._sink.sink_type != SinkType.DISCONNECTED:
                for destination in self._destinations:
                    destination.dispatch(
                        original_image=stream_data.frame_data,
                        image_with_visualization=image_with_visualization,
                        predictions=prediction,
                    )

            # Collect the image to project dataset if needed
            self._data_collector.collect(
                timestamp=stream_data.timestamp,
                frame_data=stream_data.frame_data,
                inference_data=inference_data,
            )
