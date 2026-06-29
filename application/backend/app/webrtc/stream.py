# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from loguru import logger

FALLBACK_FRAME = np.full((64, 64, 3), 16, dtype=np.uint8)

# Maximum number of short polling attempts before returning a cached/fallback frame.
# Each attempt sleeps briefly (~16ms) to avoid busy-waiting while still keeping latency low.
_MAX_POLL_ATTEMPTS = 6
_POLL_INTERVAL_S = 0.016  # ~60 Hz polling


class InferenceVideoStreamTrack(VideoStreamTrack):
    """A video stream track that provides frames with inference results over WebRTC."""

    def __init__(self, stream_queue: queue.Queue[np.ndarray]):
        super().__init__()
        self._stream_queue = stream_queue
        self._last_frame: np.ndarray | None = None

    async def recv(self) -> VideoFrame:
        """
        Asynchronously receive the next video frame from the internal queue.

        This coroutine polls ``self._stream_queue`` with short non-blocking attempts
        separated by brief asyncio sleeps (≈16 ms each, up to ~100 ms total).
        This keeps latency low while never blocking the event loop for extended periods,
        ensuring that aiortc can process ICE keep-alives and other connections concurrently.

        If a new frame is received, it is cached in ``self._last_frame``.
        If the queue is empty after all polling attempts, the method returns the
        last cached frame (if available) or ``FALLBACK_FRAME`` (a 64x64 dark gray image).

        Returns:
            aiortc.VideoFrame:
                Video frame object containing image data, presentation timestamp (``pts``),
                and time base.

        Raises:
            Exception:
                Logs and propagates any errors during retrieval or conversion.
        """
        pts, time_base = await self.next_timestamp()

        try:
            frame_data = await self._poll_frame()

            # Convert numpy array to VideoFrame
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            return frame
        except Exception:
            logger.exception("Error in recv")
            raise

    async def _poll_frame(self) -> np.ndarray:
        """Poll the stream queue with short non-blocking attempts.

        Returns the first available frame, or falls back to the last cached frame / fallback.
        """
        for _ in range(_MAX_POLL_ATTEMPTS):
            try:
                frame_data = self._stream_queue.get_nowait()
                self._last_frame = frame_data
                return frame_data
            except queue.Empty:
                await asyncio.sleep(_POLL_INTERVAL_S)

        # No frame arrived within the polling window - use fallback
        if self._last_frame is not None:
            logger.debug("No frame arrived within the polling window; reusing last frame")
            return self._last_frame
        logger.debug("No frame available yet; using fallback frame")
        return FALLBACK_FRAME
