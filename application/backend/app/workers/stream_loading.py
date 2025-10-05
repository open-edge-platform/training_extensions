# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from app.entities.stream_data import StreamData
from app.entities.video_stream import VideoStream
from app.schemas import Source, SourceType
from app.services import ActivePipelineService, VideoStreamService
from app.workers.base import BaseProcessWorker

logger = logging.getLogger(__name__)


class StreamLoader(BaseProcessWorker):
    """A process that loads frames from the video stream and injects them into the frame queue."""

    ROLE = "StreamLoader"

    def __init__(self, frame_queue: mp.Queue, stop_event: EventClass, config_changed_condition: ConditionClass) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[frame_queue])
        self._frame_queue = frame_queue
        self._config_changed_condition = config_changed_condition

        self._active_pipeline_service: ActivePipelineService | None = None
        self._prev_source_config: Source | None = None
        self._video_stream: VideoStream | None = None

    def setup(self) -> None:
        self._active_pipeline_service = ActivePipelineService(config_changed_condition=self._config_changed_condition)

    def _reset_stream_if_needed(self, source_config: Source) -> None:
        if self._prev_source_config is None or source_config != self._prev_source_config:
            logger.debug(f"Source configuration changed from {self._prev_source_config} to {source_config}")
            if self._video_stream is not None:
                self._video_stream.release()
            self._video_stream = VideoStreamService.get_video_stream(input_config=source_config)
            self._prev_source_config = copy.deepcopy(source_config)

    def run_loop(self) -> None:
        while not self.should_stop():
            source_config = self._active_pipeline_service.get_source_config()  # type: ignore

            if source_config.source_type == SourceType.DISCONNECTED:
                logger.debug("No source available... retrying in 1 second")
                self.stop_aware_sleep(1)
                continue

            self._reset_stream_if_needed(source_config)

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
