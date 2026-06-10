# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from app.models import SourceType
from app.stream.base_opencv_stream import BaseOpenCVStream
from app.stream.stream_data import StreamData


class VideoFileStream(BaseOpenCVStream):
    """Video stream implementation using video file via OpenCV."""

    def __init__(self, video_path: str, loop: bool = False) -> None:
        """Initialize video file stream.

        Args:
            video_path: Path to the video file.
            loop: If True, the video restarts from the beginning once it ends.
                  If False, the stream stops returning frames once the video has been fully read.
        """
        self.loop = loop
        self._exhausted = False
        super().__init__(source=video_path, source_type=SourceType.VIDEO_FILE, video_path=video_path, loop=loop)

    def _handle_read_failure(self) -> np.ndarray:
        """Reset video to beginning when it ends and try again (only when looping)."""
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")

        if not self.loop:
            # Mark the stream as exhausted; subsequent get_data() calls will return None.
            self._exhausted = True
            self.release()
            self.cap = None  # type: ignore[assignment]
            raise EOFError("End of video file reached")

        # Reset video to beginning when it ends
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from video file")
        return frame

    def get_data(self) -> StreamData | None:
        """Get the latest frame from the video, or None if the video has ended (non-looping)."""
        if self._exhausted:
            return None
        try:
            return super().get_data()
        except EOFError:
            return None

    def is_real_time(self) -> bool:
        return False

    def is_finished(self) -> bool:
        """A non-looping video file is finished once it has been fully read."""
        return self._exhausted
