import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import cv2
import numpy as np

from app.entities.stream_data import StreamData
from app.schemas.configuration.input_config import SourceType


class VideoStream(ABC):
    """Abstract base class for video stream implementations."""

    @abstractmethod
    def get_data(self) -> StreamData:
        """Get the latest frame from the video stream.
        Returns:
            np.ndarray: The latest frame as a numpy array
        """

    @abstractmethod
    def is_real_time(self) -> bool:
        """Check if the video stream is real-time.
        Returns:
            bool: True if the video stream is real-time, False otherwise
        """

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.release()

    @abstractmethod
    def release(self) -> None:
        """Release the video stream resources."""

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            try:
                yield self.get_data()
            except RuntimeError as exc:
                self.release()
                raise StopIteration from exc


class WebcamStream(VideoStream):
    """Video stream implementation using webcam via OpenCV."""

    def __init__(self, device_id: int = 0) -> None:
        """Initialize webcam stream.
        Args:
            device_id (int): The device ID of the webcam (default: 0)
        """
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam with device ID {device_id}")

    def get_data(self) -> StreamData:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        return StreamData(
            frame_data=frame,
            timestamp=time.time(),
            source_metadata={
                "source_type": SourceType.WEBCAM.value,
                "device_id": self.device_id,
            },
        )

    def is_real_time(self) -> bool:
        return True

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class VideoFileStream(VideoStream):
    """Video stream implementation using video file via OpenCV."""

    def __init__(self, video_path: str) -> None:
        """Initialize video file stream.
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_data(self) -> StreamData:
        ret, frame = self.cap.read()
        if not ret:
            # Reset video to beginning when it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from video file")
        return StreamData(
            frame_data=frame,
            timestamp=time.time(),
            source_metadata={
                "source_type": SourceType.VIDEO_FILE.value,
                "video_path": self.video_path,
            },
        )

    def is_real_time(self) -> bool:
        return False

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
