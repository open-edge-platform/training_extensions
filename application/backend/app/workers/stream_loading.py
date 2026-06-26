# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Condition, Lock
from multiprocessing.synchronize import Event as EventClass
from threading import Thread

from loguru import logger
from loguru._logger import Logger as LoguruLogger

from app.db import get_db_session
from app.models import DisconnectedSourceConfig, Source, SourceStatus, SourceStatusCode, SourceType
from app.services import SourceService, VideoStreamService
from app.stream.stream_data import StreamData
from app.stream.video_stream import VideoStream
from app.workers.base import BaseProcessWorker
from app.workers.shm_status import write_status


class StreamLoader(BaseProcessWorker):
    """A process that loads frames from the video stream and injects them into the frame queue."""

    ROLE = "StreamLoader"

    def __init__(
        self,
        frame_queue: mp.Queue,
        status_shm_name: str,
        status_shm_lock: Lock,
        stop_event: EventClass,
        source_changed_condition: Condition | None,
        logger_: LoguruLogger,
    ) -> None:
        super().__init__(stop_event=stop_event, logger_=logger_, queues_to_cancel=[frame_queue])
        self._frame_queue = frame_queue
        self._source_changed_condition = source_changed_condition
        self._status_shm_name = status_shm_name
        self._status_shm_lock = status_shm_lock
        self._status_shm: SharedMemory | None = None

        self._source: Source = DisconnectedSourceConfig()
        self._video_stream: VideoStream | None = None

    def _load_source(self) -> None:
        with get_db_session() as db:
            source = SourceService(db_session=db).get_active_source()
        self._source = source if source is not None else DisconnectedSourceConfig()
        logger.info(
            "Active source set to id={} name={!r} type={}. Process: {}",
            self._source.id,
            self._source.name,
            self._source.source_type,
            mp.current_process().name,
        )
        self._reset_stream()

    def _reload_source_loop(self) -> None:
        if self._source_changed_condition is None:
            return
        while True:
            with self._source_changed_condition:
                notified = self._source_changed_condition.wait(timeout=3)
                if not notified:  # awakened because of timeout
                    continue
                try:
                    self._load_source()
                except Exception:
                    logger.exception("Error reloading source")

    def setup(self) -> None:
        super().setup()
        self._status_shm = SharedMemory(name=self._status_shm_name, create=False)
        self._load_source()
        Thread(target=self._reload_source_loop, name="Source reloader", daemon=True).start()

    def _reset_stream(self) -> None:
        if self._video_stream is not None:
            self._video_stream.release()
            self._video_stream = None
        try:
            self._video_stream = VideoStreamService.get_video_stream(input_config=self._source)
        except Exception:
            logger.exception(
                "Failed to open video stream for source: id={} name={!r} type={}",
                self._source.id,
                self._source.name,
                self._source.source_type,
            )
            self._video_stream = None
            self._report_status(
                SourceStatusCode.ERROR,
                f"Failed to open video stream for source {self._source.name!r}",
            )

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
                    self._report_status(SourceStatusCode.OK)
                elif self._video_stream.is_finished():
                    # Finite source fully consumed: stop the stream instead of polling forever.
                    logger.info(
                        "Video stream finished for source id={} name={!r}; stopping stream until source changes.",
                        self._source.id,
                        self._source.name,
                    )
                    self._video_stream.release()
                    self._video_stream = None
                    self._report_status(
                        SourceStatusCode.FINISHED,
                        f"Stream finished for source {self._source.name!r}",
                    )
                else:
                    self._report_status(SourceStatusCode.OK)
                    self.stop_aware_sleep(0.1)
            except Exception:
                logger.exception("Error acquiring frame")
                self._report_status(SourceStatusCode.ERROR, "Error acquiring frame")
                self.stop_aware_sleep(2)

    def teardown(self) -> None:
        if self._video_stream is not None:
            logger.debug("Releasing video stream...")
            self._video_stream.release()
        if self._status_shm is not None:
            self._status_shm.close()

    def _report_status(self, code: SourceStatusCode, message: str = "") -> None:
        """Write the latest status into shared memory (overwrites previous value)."""
        if self._status_shm is None:
            return
        status = SourceStatus(code=code, source_id=self._source.id, message=message)
        try:
            write_status(status, self._status_shm, self._status_shm_lock)
        except Exception:
            logger.debug("Failed to write source status to shared memory")


def _enqueue_frame_with_retry(
    frame_queue: mp.Queue, payload: StreamData, is_real_time: bool, stop_event: EventClass
) -> None:
    """Enqueue a frame; for real-time sources drop the stalest queued frame
    instead of blocking the producer, so the network reader is never back-pressured.
    """
    while not stop_event.is_set():
        try:
            # For real-time sources, never block on a full queue: we must be able to
            # evict stale frames immediately so the latest frame always wins.
            if is_real_time:
                frame_queue.put_nowait(payload)
            else:
                frame_queue.put(payload, timeout=1)
            break
        except queue.Full:
            if is_real_time:
                # Drop-and-replace: discard the oldest queued frame in favour of the newest.
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(payload)
                except queue.Full:
                    logger.debug("Frame queue is full, skipping frame")
                break
            logger.debug("Frame queue is full, retrying...")
