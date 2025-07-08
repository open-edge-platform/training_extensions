import logging
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass
from typing import Any

from model_api.models import DetectionResult, Model

from app.services import ModelService
from app.utils.diagnostics import log_threads
from app.utils.visualization import Visualizer

logger = logging.getLogger(__file__)


def inference_routine(  # noqa: C901
    frame_queue: mp.Queue, pred_queue: mp.Queue, stop_event: EventClass, model_reload_event: EventClass
) -> None:
    """Load frames from the frame queue, run inference then inject the result into the predictions queue"""

    def on_inference_completed(inf_result: DetectionResult, userdata: dict[str, Any]) -> None:
        original_frame = userdata["original_frame"]
        frame_with_predictions = Visualizer.overlay_predictions(original_image=frame, predictions=inf_result)
        try:
            pred_queue.put((original_frame, frame_with_predictions, inf_result), timeout=1)  # noqa: F821
        except queue.Full:
            # TODO for non-real-time streams (e.g. video files) retry after some time instead of skipping
            #  to ensure that every frame is eventually processed
            logger.debug("Prediction queue is full; skipping")

    model_service = ModelService()
    model: Model | None = None
    last_model_id: int = 0  # track the id of the Model object to install the callback only once

    while not stop_event.is_set():
        # Get the model, reloading it if necessary
        if not model_reload_event.is_set():
            model = model_service.get_inference_model()
        else:
            # The 'while' loop handles the case when the active model is switched again while reloading.
            while model_reload_event.is_set():
                model_reload_event.clear()
                model = model_service.get_inference_model(force_reload=True)

        if model is None:
            logger.debug("No model available... retrying in 1 second")
            time.sleep(1)
            continue

        # Install the callback if it's the first iteration with this model
        if id(model) != last_model_id:
            model.set_callback(on_inference_completed)
            last_model_id = id(model)
            logger.debug(f"Installed inference callback for model object with id '{last_model_id}'")

        if model.inference_adapter.is_ready():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            model.infer_async(frame, user_data={"original_frame": frame})
        else:
            model.inference_adapter.await_any()

    logger.info("Inference routine stopped")

    # Empty the prediction queue to ensure the termination of QueueFeederThread (internal thread of 'mp.Queue')
    logger.debug("Flushing the prediction queue from leftover items")
    while pred_queue.qsize() > 0:
        pred_queue.get()
    del pred_queue

    log_threads(log_level=logging.DEBUG)
