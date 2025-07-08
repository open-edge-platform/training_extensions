# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py  (UI_MODE=True)
#  - uv run fastapi run  (UI_MODE=False)
#  - uv run fastapi dev  (UI_MODE=False, development mode)
# or use docker and access UI and backend at geti-edge.localhost
#  - docker compose up
#  - docker compose -f docker-compose.dev.yaml up

import logging
import multiprocessing as mp
import os
import signal
import threading
from collections.abc import Iterator
from pathlib import Path

import anyio
import gradio as gr
import numpy as np
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastrtc import AdditionalOutputs, Stream
from pydantic import BaseModel, Field

from app.api.endpoints import configuration, model_management
from app.utils.ipc import (
    frame_queue,
    mp_config_changed_condition,
    mp_reload_model_event,
    mp_stop_event,
    pred_queue,
    rtc_stream_queue,
)
from app.workers import dispatching_routine, frame_acquisition_routine, inference_routine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def handle_signal(signum, frame) -> None:  # noqa: ANN001, ARG001
    """Shutdown handler for SIGINT and SIGTERM."""
    logger.info(f"Process '{os.getpid()}' received signal {signum}, shutting down...")

    pid = os.getpid()
    cur_process = psutil.Process(pid)
    alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
    logger.debug(f"Alive children of process '{pid}': {alive_children}")

    mp_stop_event.set()


# Install the signal handlers before fork() so that all child processes inherit it
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

stream_loader_proc = mp.Process(
    target=frame_acquisition_routine,
    name="Stream loader",
    args=(frame_queue, mp_stop_event, mp_config_changed_condition),
)
inference_server_proc = mp.Process(
    target=inference_routine, name="Inferencer", args=(frame_queue, pred_queue, mp_stop_event, mp_reload_model_event)
)
dispatching_thread = threading.Thread(
    target=dispatching_routine,
    name="Dispatching thread",
    args=(pred_queue, rtc_stream_queue, mp_stop_event, mp_config_changed_condition),
)
stream_loader_proc.start()
inference_server_proc.start()
dispatching_thread.start()


def rtc_stream_routine() -> Iterator[tuple[np.ndarray, AdditionalOutputs]]:
    """Iterator to send frames with predictions to the WebRTC visualization stream"""
    while not mp_stop_event.is_set():
        yield rtc_stream_queue.get()
    logger.info("Stopped RTC stream routine")


stream = Stream(
    handler=rtc_stream_routine,
    modality="video",
    mode="receive",
    additional_outputs=[
        gr.Textbox(label="Predictions"),
        gr.Textbox(label="Memory Usage (MB)"),
    ],
    additional_outputs_handler=lambda _c1, _c2, pred, mem: (pred, mem),
)

app = FastAPI(
    title="GETI Edge",
    description="Edge inference server for GETI models",
    openapi_url="/api/openapi.json",
    redoc_url=None,
    docs_url=None,
    # TODO add contact info
    # TODO add license
)
app.add_middleware(  # TODO restrict settings in production
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(model_management.router)
app.include_router(configuration.router)

cur_dir = Path(__file__).parent


@app.get("/api/docs")
async def get_scalar_docs() -> HTMLResponse:
    """Shows docs for our OpenAPI specification using scalar"""
    async with await anyio.open_file(cur_dir / "scalar.html") as file:
        html_content = await file.read()
        return HTMLResponse(content=html_content)


class InputData(BaseModel):
    webrtc_id: str
    conf_threshold: float = Field(ge=0, le=1)


@app.post("/api/input_hook", tags=["webrtc"])
async def webrtc_input_hook(data: InputData) -> None:
    """Update webrtc input for user"""
    stream.set_input(data.webrtc_id, data.conf_threshold)


stream.mount(app, "/api")

if __name__ == "__main__":
    if os.getenv("GRADIO_UI") is not None:
        stream.ui.launch(server_name="0.0.0.0", server_port=7860)  # noqa: S104
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)  # noqa: S104

    # TODO send SIGKILL as a last resort if processes do not shutdown within a reasonable time
    logger.info("Joining dispatching thread...")
    dispatching_thread.join()
    logger.info("Joining inferencer...")
    inference_server_proc.join()
    logger.info("Joining stream loader...")
    stream_loader_proc.join()
    logger.info("All workers shut down gracefully.")
