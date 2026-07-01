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

    def test_is_finished_reports_true_at_eof_when_not_looping(self, video_path: str):
        """is_finished() is False until the non-looping video is fully consumed, then True."""
        with VideoFileStream(video_path=video_path, loop=False) as stream:
            # Not finished before reaching the end.
            assert stream.is_finished() is False
            while stream.get_data() is not None:
                # Still producing frames -> not finished yet.
                assert stream.is_finished() is False

            # Once exhausted, the stream reports it is finished so consumers can stop polling.
            assert stream.is_finished() is True

    def test_is_finished_stays_false_when_looping(self, video_path: str):
        """A looping stream never finishes, even after reading past the end of the file."""
        with VideoFileStream(video_path=video_path, loop=True) as stream:
            for _ in range(NUM_FRAMES * 2 + 1):
                assert stream.get_data() is not None
                assert stream.is_finished() is False

    def test_looping_can_be_re_enabled_after_a_non_looping_stream_finished(self, video_path: str):
        """After a non-looping stream is exhausted and stopped, a fresh looping stream over the
        same file works again (i.e. re-enabling looping after being disabled still works)."""
        # First, a non-looping stream that runs to completion and stops.
        with VideoFileStream(video_path=video_path, loop=False) as finite_stream:
            while finite_stream.get_data() is not None:
                pass
            assert finite_stream.is_finished() is True

        # Re-enable looping by opening a new stream over the same file; it must keep producing
        # frames indefinitely and never report itself as finished.
        with VideoFileStream(video_path=video_path, loop=True) as looping_stream:
            for _ in range(NUM_FRAMES * 2 + 1):
                data = looping_stream.get_data()
                assert data is not None
                assert data.frame_data is not None
            assert looping_stream.is_finished() is False
            assert looping_stream._exhausted is False
