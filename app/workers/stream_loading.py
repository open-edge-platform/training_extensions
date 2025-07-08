import copy
import logging
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from app.entities.video_stream import VideoStream
from app.schemas.configuration import InputConfig
from app.services import ConfigurationService, VideoStreamService
from app.utils.diagnostics import log_threads

logger = logging.getLogger(__name__)


def frame_acquisition_routine(
    frame_queue: mp.Queue, stop_event: EventClass, config_changed_condition: ConditionClass
) -> None:
    """Load frames from the video stream and inject them into the frame queue"""
    config_service = ConfigurationService(config_changed_condition=config_changed_condition)
    prev_in_config: InputConfig | None = None
    video_stream: VideoStream | None = None

    while not stop_event.is_set():
        in_config = config_service.get_input_config()

        # Reset the video stream if the configuration has changed
        if prev_in_config is None or in_config != prev_in_config:
            logger.debug(f"Input configuration changed from {prev_in_config} to {in_config}")
            if video_stream is not None:
                video_stream.release()
            video_stream = VideoStreamService.get_video_stream(input_config=in_config)
            prev_in_config = copy.deepcopy(in_config)

        if video_stream is None:
            logger.debug("No video stream available... retrying in 1 second")
            time.sleep(1)
            continue

        # Acquire a frame and enqueue it
        try:
            frame = video_stream.get_frame()
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            # TODO for non-real-time streams (e.g. video files) retry after some time instead of skipping
            #  to ensure that every frame is eventually processed
            logger.debug("Frame queue is full, skipping frame")
            continue

    logger.info("Stream acquisition stopped, releasing video stream")
    if video_stream is not None:
        video_stream.release()

    # Empty the frame queue to ensure the termination of QueueFeederThread (internal thread of 'mp.Queue')
    logger.debug("Flushing the frame queue from leftover frames")
    while frame_queue.qsize() > 0:
        frame_queue.get()
    del frame_queue

    log_threads(log_level=logging.DEBUG)
