# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models import SourceType
from app.stream.video_file_stream import VideoFileStream

NUM_FRAMES = 5
FRAME_WIDTH = 64
FRAME_HEIGHT = 48


@pytest.fixture()
def video_path(tmp_path: Path) -> Generator[str]:
    """Generate a small temporary video file with a known number of frames."""
    path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (FRAME_WIDTH, FRAME_HEIGHT))
    assert writer.isOpened(), "Failed to open VideoWriter for test fixture"
    try:
        for i in range(NUM_FRAMES):
            # Distinct solid-color frame per index so frames are clearly decodable.
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), fill_value=(i * 40) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    assert path.exists() and path.stat().st_size > 0
    yield str(path)


class TestVideoFileStream:
    """Integration tests for VideoFileStream end-of-stream semantics."""

    def test_is_not_real_time(self, video_path: str):
        with VideoFileStream(video_path=video_path) as stream:
            assert stream.is_real_time() is False

    def test_can_read_frames_and_metadata(self, video_path: str):
        """The stream yields decodable frames with the expected metadata."""
        with VideoFileStream(video_path=video_path) as stream:
            data = stream.get_data()
            assert data is not None
            frame, metadata = data.frame_data, data.source_metadata
            assert frame is not None
            assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
            assert frame.dtype == "uint8"
            assert metadata["source_type"] == SourceType.VIDEO_FILE.value
            assert metadata["video_path"] == video_path

    def test_stops_producing_frames_at_eof_when_not_looping(self, video_path: str):
        """With loop=False the stream stops producing frames once the video is exhausted."""
        with VideoFileStream(video_path=video_path, loop=False) as stream:
            frames_read = 0
            while True:
                data = stream.get_data()
                if data is None:
                    break
                assert data.frame_data is not None
                frames_read += 1
                # Guard against an accidental infinite loop in case EOF is not detected.
                assert frames_read <= NUM_FRAMES + 1

            assert frames_read == NUM_FRAMES

            # Once exhausted, the stream keeps returning None and releases the capture.
            assert stream._exhausted is True
            assert stream.cap is None
            assert stream.get_data() is None

    def test_restarts_from_beginning_when_looping(self, video_path: str):
        """With loop=True the stream restarts from frame 0 instead of stopping at EOF."""
        with VideoFileStream(video_path=video_path, loop=True) as stream:
            # Read more frames than the video contains; the stream should keep producing
            # frames by looping back to the start.
            for _ in range(NUM_FRAMES * 2 + 1):
                data = stream.get_data()
                assert data is not None
                assert data.frame_data is not None

            assert stream._exhausted is False
