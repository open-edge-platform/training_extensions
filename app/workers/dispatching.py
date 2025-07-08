import copy
import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from fastrtc import AdditionalOutputs

from app.entities.dispatchers import Dispatcher
from app.schemas.configuration import OutputConfig
from app.services import ConfigurationService, DispatchService, SystemService

logger = logging.getLogger(__name__)


def dispatching_routine(
    pred_queue: mp.Queue,
    rtc_stream_queue: queue.Queue,
    stop_event: EventClass,
    config_changed_condition: ConditionClass,
) -> None:
    """Pull predictions from the queue and dispatch them to the configured outputs and WebRTC visualization stream."""
    system_service = SystemService()
    config_service = ConfigurationService(config_changed_condition=config_changed_condition)

    prev_out_config: list[OutputConfig] = []
    destinations: list[Dispatcher] = []

    while not stop_event.is_set():
        out_config = config_service.get_output_config()

        if not prev_out_config or out_config != prev_out_config:
            logger.debug(f"Output config changed from {prev_out_config} to {out_config}")
            destinations = DispatchService.get_destinations(output_configs=out_config)
            prev_out_config = copy.deepcopy(out_config)

        # Read from the queue
        try:
            original_frame, frame_with_detections, inf_result = pred_queue.get(timeout=1)
        except queue.Empty:
            logger.debug("Nothing to dispatch yet")
            continue

        # Postprocess and dispatch results
        for destination in destinations:
            destination.dispatch(
                original_image=original_frame,
                image_with_visualization=frame_with_detections,
                predictions=inf_result,
            )

        # Dispatch to WebRTC stream
        mem_mb, _ = system_service.get_memory_usage()
        additional_outputs = AdditionalOutputs(str(inf_result), f"{mem_mb:.2f} MB")
        try:
            rtc_stream_queue.put((frame_with_detections, additional_outputs), block=False)
        except queue.Full:
            logger.debug("Visualization queue is full; skipping")

    logger.info("Stopped dispatching routine")
