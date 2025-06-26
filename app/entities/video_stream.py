from abc import ABC, abstractmethod
from collections.abc import Iterator

import cv2
import numpy as np


class VideoStream(ABC):
    """Abstract base class for video stream implementations."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Get the latest frame from the video stream.
        Returns:
            np.ndarray: The latest frame as a numpy array
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
                yield self.get_frame()
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
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam with device ID {device_id}")

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        return frame

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
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            # Reset video to beginning when it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from video file")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
