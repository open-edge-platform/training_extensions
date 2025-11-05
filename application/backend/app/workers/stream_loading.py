# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition
from multiprocessing.synchronize import Event as EventClass
from threading import Thread

from app.db import get_db_session
from app.models import DisconnectedSourceConfig, Source, SourceType
from app.services import SourceService, VideoStreamService
from app.stream.stream_data import StreamData
from app.stream.video_stream import VideoStream
from app.workers.base import BaseProcessWorker

logger = logging.getLogger(__name__)


class StreamLoader(BaseProcessWorker):
    """A process that loads frames from the video stream and injects them into the frame queue."""

    ROLE = "StreamLoader"

    def __init__(
        self,
        frame_queue: mp.Queue,
        stop_event: EventClass,
        source_changed_condition: Condition | None,
    ) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[frame_queue])
        self._frame_queue = frame_queue
        self._source_changed_condition = source_changed_condition

        self._source: Source = DisconnectedSourceConfig()
        self._video_stream: VideoStream | None = None

    def _load_source(self) -> None:
        with get_db_session() as db:
            source = SourceService(db_session=db).get_active_source()
        self._source = source if source is not None else DisconnectedSourceConfig()
        logger.info(f"Active source set to {self._source}. Process: %s", mp.current_process().name)
        self._reset_stream()

    def _reload_source_loop(self) -> None:
        if self._source_changed_condition is None:
            return
        while True:
            with self._source_changed_condition:
                notified = self._source_changed_condition.wait(timeout=3)
                if not notified:  # awakened because of timeout
                    continue
                self._load_source()

    def setup(self) -> None:
        self._load_source()
        Thread(target=self._reload_source_loop, name="Source reloader", daemon=True).start()

    def _reset_stream(self) -> None:
        if self._video_stream is not None:
            self._video_stream.release()
        self._video_stream = VideoStreamService.get_video_stream(input_config=self._source)

    def run_loop(self) -> None:
        while not self.should_stop():
            if self._source.source_type == SourceType.DISCONNECTED:
                logger.debug("No source available... retrying in 1 second")
                self.stop_aware_sleep(1)
                continue

            if self._video_stream is None:
                logger.debug("No video stream available, retrying in 1 second...")
                self.stop_aware_sleep(1)
                continue

            # Acquire a frame and enqueue it
            try:
                stream_data = self._video_stream.get_data()
                if stream_data is not None:
                    _enqueue_frame_with_retry(
                        self._frame_queue, stream_data, self._video_stream.is_real_time(), self._stop_event
                    )
                else:
                    self.stop_aware_sleep(0.1)
            except Exception:
                logger.exception("Error acquiring frame")
                self.stop_aware_sleep(2)

    def teardown(self) -> None:
        if self._video_stream is not None:
            logger.debug("Releasing video stream...")
            self._video_stream.release()


def _enqueue_frame_with_retry(
    frame_queue: mp.Queue, payload: StreamData, is_real_time: bool, stop_event: EventClass
) -> None:
    """Enqueue frame with retry logic for non-real-time streams"""
    while not stop_event.is_set():
        try:
            frame_queue.put(payload, timeout=1)
            break
        except queue.Full:
            if is_real_time:
                logger.debug("Frame queue is full, skipping frame")
                break
            logger.debug("Frame queue is full, retrying...")
