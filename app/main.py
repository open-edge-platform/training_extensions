# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run fastapi run
#  - uv run fastapi dev
# or use docker and access UI and backend at geti-edge.localhost
#  - docker build -t geti-edge .
#  - docker run --network host --name geti-edge geti-edge

import time
from collections.abc import Generator
from pathlib import Path

import anyio
import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastrtc import AdditionalOutputs, Stream
from pydantic import BaseModel, Field

from app.api.endpoints import model_management
from app.services import ModelService, SystemService
from app.video_stream import VideoFileStream, VideoStream, WebcamStream
from app.visualization import DetectionVisualizer

VIDEO_PATH = "data/media/video.mp4"
UI_MODE: bool = False
MOCK_CAMERA: bool = True


def get_video_stream() -> VideoStream:
    """Initialize the camera"""
    if MOCK_CAMERA:
        return VideoFileStream(VIDEO_PATH)
    return WebcamStream()


def acquire_and_detect() -> Generator[tuple[np.ndarray, AdditionalOutputs], None, None]:
    """Main loop. Acquires frames from the video stream and detects objects in them."""
    video_stream = iter(get_video_stream())
    model_service = ModelService()
    system_service = SystemService()

    while True:
        model = model_service.get_inference_model()
        if not model:
            time.sleep(1)
            continue
        frame = next(video_stream)
        detections = model(frame)
        frame_with_detections = DetectionVisualizer.overlay_predictions(original_image=frame, predictions=detections)
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
    if UI_MODE:
        stream.ui.launch(server_name="0.0.0.0", server_port=7860)  # noqa: S104
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)  # noqa: S104
