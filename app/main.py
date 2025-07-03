# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py  (UI_MODE=True)
#  - uv run fastapi run  (UI_MODE=False)
#  - uv run fastapi dev  (UI_MODE=False, development mode)
# or use docker and access UI and backend at geti-edge.localhost
#  - docker compose up
#  - docker compose -f docker-compose.dev.yaml up

import copy
import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastrtc import AdditionalOutputs, Stream
from pydantic import BaseModel, Field

from app.api.endpoints import configuration, model_management
from app.entities.dispatchers import Dispatcher
from app.services import ConfigurationService, DispatchService, ModelService, SystemService, VideoStreamService
from app.visualization import DetectionVisualizer

if TYPE_CHECKING:
    from schemas.configuration import AppConfig

    from app.entities.video_stream import VideoStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def acquire_and_detect() -> Generator[tuple[np.ndarray, AdditionalOutputs], None, None]:
    """Main loop. Acquires frames from the video stream and detects objects in them."""
    model_service = ModelService()
    system_service = SystemService()
    config_service = ConfigurationService()

    prev_config: AppConfig | None = None
    video_stream: VideoStream | None = None
    destinations: list[Dispatcher] = []

    while True:
        # Get the latest configuration
        config = config_service.get_app_config()

        # Reset the video stream if the input configuration has changed
        if prev_config is None or config.input != prev_config.input:
            logger.debug(
                f"Input configuration changed from {prev_config.input if prev_config else 'None'} to {config.input}"
            )
            if video_stream is not None:
                video_stream.release()
            video_stream = VideoStreamService.get_video_stream(input_config=config.input)

        if prev_config is None or config.outputs != prev_config.outputs:
            logger.debug(
                f"Output config changed from {prev_config.outputs if prev_config else 'None'} to {config.outputs}"
            )
            destinations = DispatchService.get_destinations(output_configs=config.outputs)

        if prev_config is None or config.input != prev_config.input or config.outputs != prev_config.outputs:
            prev_config = copy.deepcopy(config)

        if video_stream is None:
            logger.debug("No video stream available... retrying in 1 second")
            time.sleep(1)
            continue

        # Get the model to use for inference
        model = model_service.get_inference_model()
        if model is None:
            logger.debug("No model available... retrying in 1 second")
            time.sleep(1)
            continue

        # Get a frame from the video stream
        frame = video_stream.get_frame()

        # Run inference
        detections = model(frame)

        # Postprocess and dispatch results
        frame_with_detections = DetectionVisualizer.overlay_predictions(original_image=frame, predictions=detections)
        for destination in destinations:
            destination.dispatch(
                original_image=frame,
                image_with_visualization=frame_with_detections,
                predictions=detections,
            )

        mem_mb, _ = system_service.get_memory_usage()
        yield frame_with_detections, AdditionalOutputs(str(detections), f"{mem_mb:.2f} MB")


stream = Stream(
    handler=acquire_and_detect,
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
