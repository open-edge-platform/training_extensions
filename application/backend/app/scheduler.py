# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import os
import queue
import threading
from multiprocessing.shared_memory import SharedMemory

import psutil

from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.metrics_service import SIZE
from app.workers import DispatchingWorker, InferenceWorker, StreamLoader

logger = logging.getLogger(__name__)


class Scheduler:
    """Manages application processes and threads"""

    FRAME_QUEUE_SIZE = 5
    PREDICTION_QUEUE_SIZE = 5

    def __init__(self, event_bus: EventBus, data_collector: DataCollector) -> None:
        self._event_bus = event_bus
        self._data_collector = data_collector

        logger.info("Initializing Scheduler...")
        # Queue for the frames acquired from the stream source and decoded
        self.frame_queue: mp.Queue = mp.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        # Queue for the inference results (predictions)
        self.pred_queue: mp.Queue = mp.Queue(maxsize=self.PREDICTION_QUEUE_SIZE)
        # Queue for pushing predictions to the visualization stream (WebRTC)
        self.rtc_stream_queue: queue.Queue = queue.Queue(maxsize=1)
        # Event to sync all processes on application shutdown
        self.mp_stop_event = mp.Event()
        # Event to signal that the model has to be reloaded
        self.mp_model_reload_event = mp.Event()

        # Shared memory for metrics collector
        self.shm_metrics = SharedMemory(create=True, size=SIZE)
        self.shm_metrics_lock = mp.Lock()

        self.processes: list[mp.Process] = []
        self.threads: list[threading.Thread] = []
        logger.info("Scheduler initialized")

    def start_workers(self) -> None:
        """Start all worker processes and threads"""
        logger.info("Starting worker processes...")

        # Create and start processes
        stream_loader_proc = StreamLoader(
            self.frame_queue, self.mp_stop_event, self._event_bus.source_changed_condition
        )

        inference_server_proc = InferenceWorker(
            frame_queue=self.frame_queue,
            pred_queue=self.pred_queue,
            stop_event=self.mp_stop_event,
            model_reload_event=self.mp_model_reload_event,
            shm_name=self.shm_metrics.name,
            shm_lock=self.shm_metrics_lock,
        )

        dispatching_thread = DispatchingWorker(
            event_bus=self._event_bus,
            pred_queue=self.pred_queue,
            rtc_stream_queue=self.rtc_stream_queue,
            stop_event=self.mp_stop_event,
            data_collector=self._data_collector,
        )

        # Start all workers
        stream_loader_proc.start()
        inference_server_proc.start()
        dispatching_thread.start()

        # Track processes and threads
        self.processes.extend([stream_loader_proc, inference_server_proc])
        self.threads.append(dispatching_thread)

        logger.info("All worker processes started successfully")

    def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")

        # Signal all processes to stop
        self.mp_stop_event.set()

        # Get current process info for debugging
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        # Join threads first
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate within timeout")

        # Join processes in reverse order so that consumers are terminated before producers.
        for process in self.processes[::-1]:
            if process.is_alive():
                logger.debug(f"Joining process: {process.name}")
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning("Force terminating process: %s", process.name)
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.error("Force killing process %s", process.name)
                        process.kill()

        logger.info("All workers shut down gracefully")

        # Clear references
        self.processes.clear()
        self.threads.clear()
        self.shm_metrics.close()
        self.shm_metrics.unlink()

        self._cleanup_queues()

    def _cleanup_queues(self) -> None:
        """Final queue cleanup"""
        for q, name in [(self.frame_queue, "frame_queue"), (self.pred_queue, "pred_queue")]:
            if q is not None:
                try:
                    # https://runebook.dev/en/articles/python/library/multiprocessing/multiprocessing.Queue.close
                    q.close()
                    q.join_thread()
                    logger.debug("Successfully cleaned up %s", name)
                except Exception as e:
                    logger.warning("Error cleaning up %s: %s", name, e)
