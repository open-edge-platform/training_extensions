# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py
# or
#  - docker build -t geti-edge .
#  - docker run --network host --name geti-edge geti-edge

import time
from collections.abc import Generator

import gradio as gr
import numpy as np
from api.endpoints import model_management
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import AdditionalOutputs, Stream
from services import ModelService, SystemService
from video_stream import VideoFileStream, VideoStream, WebcamStream
from visualization import DetectionVisualizer

VIDEO_PATH = "data/media/video.mp4"
UI_MODE: bool = True
MOCK_CAMERA: bool = True


def init_video_stream() -> VideoStream:
    """Initialize the camera"""
    if MOCK_CAMERA:
        return VideoFileStream(VIDEO_PATH)
    return WebcamStream()


def acquire_and_detect() -> Generator[tuple[np.ndarray, AdditionalOutputs], None, None]:
    """Main loop. Acquires frames from the video stream and detects objects in them."""
    video_stream = init_video_stream()
    model_service = ModelService()
    system_service = SystemService()
    while True:
        model = model_service.get_inference_model()
        if not model:
            time.sleep(1)
            continue
        frame = video_stream.get_frame()
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

stream.mount(app)

if __name__ == "__main__":
    if UI_MODE:
        stream.ui.launch(server_name="0.0.0.0", server_port=7860)  # noqa: S104
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)  # noqa: S104
