import copy
import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from fastrtc import AdditionalOutputs

from app.entities.stream_data import StreamData
from app.schemas.configuration import Sink
from app.services import ConfigurationService, DispatchService
from app.services.dispatchers import Dispatcher

logger = logging.getLogger(__name__)


def dispatching_routine(
    pred_queue: mp.Queue,
    rtc_stream_queue: queue.Queue,
    stop_event: EventClass,
    config_changed_condition: ConditionClass,
) -> None:
    """Pull predictions from the queue and dispatch them to the configured outputs and WebRTC visualization stream."""
    config_service = ConfigurationService(config_changed_condition=config_changed_condition)

    prev_sink_config: list[Sink] = []
    destinations: list[Dispatcher] = []

    while not stop_event.is_set():
        sink_config = config_service.get_sink_config()

        if not prev_sink_config or sink_config != prev_sink_config:
            logger.debug(f"Sink config changed from {prev_sink_config} to {sink_config}")
            destinations = DispatchService.get_destinations(output_configs=[sink_config])
            prev_sink_config = copy.deepcopy(sink_config)

        # Read from the queue
        try:
            stream_data: StreamData = pred_queue.get(timeout=1)
        except queue.Empty:
            logger.debug("Nothing to dispatch yet")
            continue

        if stream_data.inference_data is None:
            logger.error("Missing inference data in stream_data; skipping dispatch")
            continue

        image_with_visualization = stream_data.inference_data.visualized_prediction
        prediction = stream_data.inference_data.prediction
        # Postprocess and dispatch results
        for destination in destinations:
            destination.dispatch(
                original_image=stream_data.frame_data,
                image_with_visualization=image_with_visualization,
                predictions=prediction,
            )

        # Dispatch to WebRTC stream
        additional_outputs = AdditionalOutputs(str(prediction))
        try:
            rtc_stream_queue.put((image_with_visualization, additional_outputs), block=False)
        except queue.Full:
            logger.debug("Visualization queue is full; skipping")

    logger.info("Stopped dispatching routine")
