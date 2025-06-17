# export no_proxy="localhost, 127.0.0.1, ::1"
# uv run app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import Stream
from model_api.models import DetectionModel

from video_stream import VideoStream, VideoFileStream, WebcamStream
from visualization import DetectionVisualizer

VIDEO_PATH = "data/media/video.mp4"
MODEL_PATH = "data/models/atss.xml"
UI_MODE: bool = True
MOCK_CAMERA: bool = True


def init_camera() -> VideoStream:
    if MOCK_CAMERA:
        return VideoFileStream(VIDEO_PATH)
    else:
        return WebcamStream()


def acquire_and_detect():
    camera = init_camera()
    detection_model = DetectionModel.create_model(MODEL_PATH)
    while True:
        frame = camera.get_frame()
        detections = detection_model(frame)
        frame_with_detections = DetectionVisualizer.overlay_predictions(
            original_image=frame, predictions=detections
        )
        yield frame_with_detections


stream = Stream(
    handler=acquire_and_detect,
    modality="video",
    mode="receive",
)

app = FastAPI()
# TODO restrict settings in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream.mount(app)

if __name__ == "__main__":
    if UI_MODE:
        stream.ui.launch(server_name="0.0.0.0", server_port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)
