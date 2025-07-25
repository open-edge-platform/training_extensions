import copy
import logging
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from app.entities.stream_data import StreamData
from app.entities.video_stream import VideoStream
from app.schemas.configuration import Source
from app.services import ConfigurationService, VideoStreamService
from app.utils.diagnostics import log_threads

logger = logging.getLogger(__name__)


def frame_acquisition_routine(
    frame_queue: mp.Queue, stop_event: EventClass, config_changed_condition: ConditionClass, cleanup: bool = True
) -> None:
    """Load frames from the video stream and inject them into the frame queue"""
    config_service = ConfigurationService(config_changed_condition=config_changed_condition)
    prev_source_config: Source | None = None
    video_stream: VideoStream | None = None

    while not stop_event.is_set():
        source_config = config_service.get_source_config()

        # Reset the video stream if the configuration has changed
        if prev_source_config is None or source_config != prev_source_config:
            logger.debug(f"Source configuration changed from {prev_source_config} to {source_config}")
            if video_stream is not None:
                video_stream.release()
            video_stream = VideoStreamService.get_video_stream(input_config=source_config)
            prev_source_config = copy.deepcopy(source_config)

        if video_stream is None:
            logger.debug("No video stream available, retrying in 1 second...")
            time.sleep(1)
            continue

        # Acquire a frame and enqueue it
        try:
            stream_data = video_stream.get_data()
            if stream_data is not None:
                _enqueue_frame_with_retry(frame_queue, stream_data, video_stream.is_real_time(), stop_event)
            else:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error acquiring frame: {e}")
            continue

    if cleanup:
        _cleanup_resources(frame_queue, video_stream)


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


def _cleanup_resources(frame_queue: mp.Queue, video_stream: VideoStream | None) -> None:
    """Clean up video stream and frame queue resources"""
    logger.info("Stream acquisition stopped, releasing video stream")
    if video_stream is not None:
        video_stream.release()

    # Empty the frame queue to ensure the termination of QueueFeederThread (internal thread of 'mp.Queue')
    logger.debug("Flushing the frame queue from leftover frames")
    while frame_queue.qsize() > 0:
        frame_queue.get()
    del frame_queue

    log_threads(log_level=logging.DEBUG)
