# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time

import cv2
import numpy as np
from loguru import logger

from app.models import IPCameraSourceConfig, SourceType
from app.stream.base_opencv_stream import BaseOpenCVStream
from app.stream.stream_data import StreamData

# Force TCP transport and low-latency demux/decode to avoid UDP packet loss
# and reordering delays that break H.264 decoding ("co located POCs unavailable").
_RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|reorder_queue_size;0|stimeout;5000000"
)

# Maximum dimension (longest edge) for frames passed to inference.
# Frames larger than this are downscaled to reduce CPU/memory pressure.
_MAX_FRAME_DIMENSION = int(os.environ.get("GETI_MAX_FRAME_DIMENSION", "1920"))


def _apply_ffmpeg_capture_options() -> None:
    """Set OpenCV's FFmpeg capture options before constructing VideoCapture."""
    existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if existing == _RTSP_FFMPEG_OPTIONS:
        return
    if existing and "rtsp_transport" in existing:
        # Respect operator overrides.
        return
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_FFMPEG_OPTIONS


class IPCameraStream(BaseOpenCVStream):
    """IP camera stream that drains the RTSP socket on a dedicated reader thread
    and exposes only the latest decoded frame, so slow consumers (e.g. inference)
    cannot back-pressure the network reader and cause upstream packet drops.
    """

    MAX_RECONNECT_ATTEMPTS = 10
    BACKOFF_FACTOR = 0.5

    _FIRST_FRAME_TIMEOUT_S = 10.0
    _NEW_FRAME_TIMEOUT_S = 1.0

    def __init__(self, config: IPCameraSourceConfig) -> None:
        stream_url = config.config_data.get_configured_stream_url()
        if str(stream_url).lower().startswith("rtsp"):
            _apply_ffmpeg_capture_options()
        else:
            # Clear RTSP-specific options that would interfere with HTTP/other streams
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        super().__init__(
            source=config.config_data.get_configured_stream_url(),
            source_type=SourceType.IP_CAMERA,
            stream_url=config.config_data.stream_url,  # Original stream URL is kept for metadata
        )
        logger.info("IP camera stream initialized")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame_lock = threading.Lock()
        self._frame_available = threading.Condition(self._frame_lock)
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float = 0.0
        self._latest_seq: int = 0
        self._last_consumed_seq: int = 0
        self._reader_error: BaseException | None = None

        self._stop_reader = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"IPCameraReader[{self.source_type.value}]",
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:  # noqa: C901
        """Continuously drain the RTSP socket and publish the latest frame."""
        consecutive_grab_failures = 0
        while not self._stop_reader.is_set():
            try:
                cap = self.cap
                if cap is None or not cap.isOpened():
                    if self._stop_reader.is_set():
                        break
                    if not self._reconnect():
                        return
                    consecutive_grab_failures = 0
                    continue

                if self._stop_reader.is_set():
                    break

                if not cap.grab():
                    consecutive_grab_failures += 1
                    if consecutive_grab_failures >= 3:
                        if not self._reconnect():
                            return
                        consecutive_grab_failures = 0
                    else:
                        time.sleep(0.05 * consecutive_grab_failures)
                    continue

                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue

                consecutive_grab_failures = 0
                self._publish_frame(frame)
            except Exception as exc:
                if self._stop_reader.is_set():
                    break
                logger.exception("IP camera reader thread error")
                with self._frame_available:
                    self._reader_error = exc
                    self._frame_available.notify_all()
                time.sleep(self.BACKOFF_FACTOR)

    def _reconnect(self) -> bool:
        """Try to re-open the capture with exponential backoff.

        Returns:
            True if reconnection succeeded, False if all attempts exhausted (gives up permanently).
        """
        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            if self._stop_reader.is_set():
                return False
            logger.warning("IP camera reconnect attempt {}/{}", attempt + 1, self.MAX_RECONNECT_ATTEMPTS)
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                logger.debug("Error releasing capture during reconnect", exc_info=True)
            try:
                self._initialize_capture()
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self.cap.isOpened():
                    logger.info("Successfully reconnected to IP camera stream.")
                    return True
            except Exception:
                logger.debug("Reconnect attempt {} failed", attempt + 1, exc_info=True)
            self._stop_reader.wait(self.BACKOFF_FACTOR * (attempt + 1))

        logger.error("IP camera unreachable after {} attempts, giving up.", self.MAX_RECONNECT_ATTEMPTS)
        with self._frame_available:
            self._reader_error = RuntimeError(
                "IP camera stream permanently unavailable after repeated reconnect failures"
            )
            self._frame_available.notify_all()
        return False

    @staticmethod
    def _downscale_if_needed(frame: np.ndarray) -> np.ndarray:
        """Downscale frame if it exceeds _MAX_FRAME_DIMENSION on either axis."""
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= _MAX_FRAME_DIMENSION:
            return frame
        scale = _MAX_FRAME_DIMENSION / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _publish_frame(self, frame: np.ndarray) -> None:
        frame = self._downscale_if_needed(frame)
        with self._frame_available:
            self._latest_frame = frame
            self._latest_timestamp = time.time()
            self._latest_seq += 1
            self._reader_error = None
            self._frame_available.notify_all()

    def get_data(self) -> StreamData:
        """Return the latest decoded frame, blocking briefly if necessary."""
        with self._frame_available:
            if self._latest_seq == 0:
                self._frame_available.wait(timeout=self._FIRST_FRAME_TIMEOUT_S)
            elif self._latest_seq == self._last_consumed_seq:
                self._frame_available.wait(timeout=self._NEW_FRAME_TIMEOUT_S)

            if self._reader_error is not None and self._latest_seq == 0:
                err = self._reader_error
                self._reader_error = None
                raise err

            if self._latest_frame is None or self._latest_seq == self._last_consumed_seq:
                raise RuntimeError("No new frame available from IP camera")

            frame = self._latest_frame
            timestamp = self._latest_timestamp
            self._last_consumed_seq = self._latest_seq

        return StreamData(
            frame_data=frame,
            timestamp=timestamp,
            source_metadata=self._get_source_metadata(),
        )

    def _handle_read_failure(self) -> np.ndarray:
        """Legacy synchronous reconnect path, kept for BaseOpenCVStream compatibility."""
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")

        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            logger.warning(f"Attempt {attempt + 1}: Failed to capture frame from IP camera, retrying...")
            ret, frame = self.cap.read()
            if ret:
                logger.info("Successfully reconnected to IP camera stream.")
                return frame
            # Reconnect before next attempt
            self.release_capture_only()
            self._initialize_capture()
            time.sleep(self.BACKOFF_FACTOR * (attempt + 1))

        raise RuntimeError("Failed to capture frame from IP camera after multiple retries")

    def release_capture_only(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                logger.debug("Error releasing VideoCapture", exc_info=True)

    def release(self) -> None:
        self._stop_reader.set()
        with self._frame_available:
            self._frame_available.notify_all()
        if self._reader_thread.is_alive() and threading.current_thread() is not self._reader_thread:
            self._reader_thread.join(timeout=5.0)
        # Only release the capture AFTER the reader thread has stopped to avoid
        # segfaults from concurrent cv2.VideoCapture access (not thread-safe).
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                logger.debug("Error releasing VideoCapture during cleanup", exc_info=True)
            self.cap = None  # type: ignore[assignment]

    def is_real_time(self) -> bool:
        return True

    def __enter__(self) -> "IPCameraStream":
        return self
