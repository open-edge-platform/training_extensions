import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import types

import psutil

from app.utils.ipc import mp_config_changed_condition, mp_reload_model_event, mp_stop_event
from app.utils.singleton import Singleton
from app.workers import dispatching_routine, frame_acquisition_routine, inference_routine

logger = logging.getLogger(__name__)


class Scheduler(metaclass=Singleton):
    """Manages application processes and threads"""

    FRAME_QUEUE_SIZE = 5
    PREDICTION_QUEUE_SIZE = 5

    def __init__(self) -> None:
        # Queue for the frames acquired from the stream source and decoded
        self.frame_queue: mp.Queue = mp.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        # Queue for the inference results (predictions)
        self.pred_queue: mp.Queue = mp.Queue(maxsize=self.PREDICTION_QUEUE_SIZE)
        # Queue for pushing predictions to the visualization stream (WebRTC)
        self.rtc_stream_queue: queue.Queue = queue.Queue(maxsize=1)

        self.processes: list[mp.Process] = []
        self.threads: list[threading.Thread] = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""

        def handle_signal(signum: int, _: types.FrameType | None) -> None:
            logger.info(f"Process '{os.getpid()}' received signal {signum}, shutting down...")
            self.shutdown()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def start_workers(self) -> None:
        """Start all worker processes and threads"""
        logger.info("Starting worker processes...")

        # Create and start processes
        stream_loader_proc = mp.Process(
            target=frame_acquisition_routine,
            name="Stream loader",
            args=(self.frame_queue, mp_stop_event, mp_config_changed_condition),
        )

        inference_server_proc = mp.Process(
            target=inference_routine,
            name="Inferencer",
            args=(self.frame_queue, self.pred_queue, mp_stop_event, mp_reload_model_event),
        )

        dispatching_thread = threading.Thread(
            target=dispatching_routine,
            name="Dispatching thread",
            args=(self.pred_queue, self.rtc_stream_queue, mp_stop_event, mp_config_changed_condition),
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
        mp_stop_event.set()

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
