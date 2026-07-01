# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import time
from threading import Thread
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.models import SourceType, VideoFileSourceConfig
from app.models.source import VideoFileConfig
from app.stream.stream_data import StreamData
from app.workers import StreamLoader


class _FakeVideoStream:
    """Minimal picklable stand-in for VideoStream."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def get_data(self) -> StreamData:
        return StreamData(frame_data=self._frame, timestamp=time.time(), source_metadata={})

    def is_real_time(self) -> bool:
        return True

    def release(self) -> None:
        pass


class _TestableStreamLoader(StreamLoader):
    def __init__(self, frame_queue, stop_event, source_changed_condition, fake_video_stream):
        super().__init__(
            frame_queue=frame_queue,
            stop_event=stop_event,
            source_changed_condition=source_changed_condition,
            logger_=Mock(),  # Mock is only used in the parent; child re-creates via super().__init__
        )
        self._fake_video_stream = fake_video_stream

    def setup(self) -> None:
        self._source = VideoFileSourceConfig(
            source_type=SourceType.VIDEO_FILE,
            id=uuid4(),
            name="Test Video File Source",
            config_data=VideoFileConfig(video_path="fake_path.mp4"),
        )
        self._video_stream = self._fake_video_stream


@pytest.fixture(scope="module")
def mp_ctx():
    """Use the same 'spawn' context as production code."""
    return mp.get_context("spawn")


@pytest.fixture
def mp_manager(mp_ctx):
    manager = mp_ctx.Manager()
    yield manager
    manager.shutdown()


@pytest.fixture
def frame_queue(mp_manager):
    return mp_manager.Queue(maxsize=2)


@pytest.fixture
def stop_event(request, mp_ctx):
    event = mp_ctx.Event()
    request.addfinalizer(event.set)
    return event


@pytest.fixture
def source_changed_condition(mp_ctx):
    return mp_ctx.Condition()


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_video_stream(sample_frame) -> _FakeVideoStream:
    return _FakeVideoStream(frame=sample_frame)


class TestStreamLoader:
    def _start_loader(self, request, frame_queue, stop_event, source_changed_condition, fake_video_stream):
        """Helper that starts a _TestableStreamLoader and registers cleanup."""
        process = _TestableStreamLoader(frame_queue, stop_event, source_changed_condition, fake_video_stream)
        process.start()

        def cleanup():
            stop_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        request.addfinalizer(cleanup)
        return process

    def test_queue_full(
        self, request, frame_queue, stop_event, source_changed_condition, fake_video_stream, sample_frame
    ):
        """Test that for real-time sources the loader drops stale frames and replaces with fresh ones."""
        sentinel = np.zeros_like(sample_frame)
        data1 = StreamData(frame_data=sentinel.copy(), timestamp=time.time(), source_metadata={"src": "pre1"})
        data2 = StreamData(frame_data=sentinel.copy(), timestamp=time.time(), source_metadata={"src": "pre2"})
        frame_queue.put(data1)
        frame_queue.put(data2)

        process = self._start_loader(request, frame_queue, stop_event, source_changed_condition, fake_video_stream)

        time.sleep(1)

        stop_event.set()
        process.join(timeout=3)

        queue_contents = []
        while not frame_queue.empty():
            try:
                queue_contents.append(frame_queue.get_nowait())
            except queue.Empty:
                break

        assert len(queue_contents) == 2
        # Both pre-filled sentinels must have been evicted in favour of fresh frames
        # produced by the loader; otherwise the drop-and-replace policy has regressed.
        assert all(np.array_equal(el.frame_data, sample_frame) for el in queue_contents), (
            "Expected every stale sentinel frame to be replaced with a fresh one"
        )
        assert not any(np.array_equal(el.frame_data, sentinel) for el in queue_contents), (
            "No sentinel frames should remain in the queue"
        )
        assert not process.is_alive(), "Process should terminate cleanly"

    def test_queue_empty(self, request, frame_queue, stop_event, source_changed_condition, fake_video_stream):
        """Test that stream frames are acquired when queue is empty."""
        process = self._start_loader(request, frame_queue, stop_event, source_changed_condition, fake_video_stream)

        # Poll until the queue reaches the expected size (maxsize=2), with a timeout of 5 seconds
        deadline = time.monotonic() + 5
        while frame_queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        stop_event.set()
        process.join(timeout=3)

        assert frame_queue.qsize() == 2
        assert not process.is_alive(), "Process should terminate cleanly"


class TestStreamLoaderErrorHandling:
    """Tests for StreamLoader resilience when video stream fails to open."""

    def _make_loader(self, frame_queue, stop_event, source_changed_condition):
        return StreamLoader(
            frame_queue=frame_queue,
            stop_event=stop_event,
            source_changed_condition=source_changed_condition,
            logger_=Mock(),
        )

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_reset_stream_handles_runtime_error(self, mock_get_stream, frame_queue, stop_event):
        """_reset_stream should catch RuntimeError and set _video_stream to None."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source: rtsp://bad")
        loader = self._make_loader(frame_queue, stop_event, None)
        loader._reset_stream()
        assert loader._video_stream is None

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_reset_stream_releases_existing_stream_on_error(self, mock_get_stream, frame_queue, stop_event):
        """_reset_stream should release the old stream even when the new one fails."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source")
        loader = self._make_loader(frame_queue, stop_event, None)
        old_stream = Mock()
        loader._video_stream = old_stream
        loader._reset_stream()
        old_stream.release.assert_called_once()
        assert loader._video_stream is None

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    @patch("app.workers.stream_loading.get_db_session")
    def test_reload_source_loop_survives_error(self, mock_db, mock_get_stream, mp_ctx, frame_queue, stop_event):
        """_reload_source_loop should not crash when _load_source raises."""
        mock_db.return_value.__enter__ = Mock(return_value=Mock())
        mock_db.return_value.__exit__ = Mock(return_value=False)
        mock_get_stream.side_effect = RuntimeError("unreachable")

        condition = mp_ctx.Condition()
        loader = self._make_loader(frame_queue, stop_event, condition)

        with patch("app.workers.stream_loading.SourceService") as mock_source_service:
            mock_source_service.return_value.get_active_source.return_value = VideoFileSourceConfig(
                source_type=SourceType.VIDEO_FILE,
                id=uuid4(),
                name="Test",
                config_data=VideoFileConfig(video_path="fake.mp4"),
            )

            thread = Thread(target=loader._reload_source_loop, daemon=True)
            thread.start()

            # Wait until get_video_stream is called (proves _load_source ran)
            deadline = time.monotonic() + 5
            while not mock_get_stream.called and time.monotonic() < deadline:
                with condition:
                    condition.notify_all()
                time.sleep(0.05)

            # Verify _load_source was actually invoked
            assert mock_get_stream.called, "_load_source should have been called"
            mock_source_service.return_value.get_active_source.assert_called()

            # Thread should still be alive (exception was caught, loop continues)
            assert thread.is_alive()

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    @patch("app.workers.stream_loading.get_db_session")
    def test_setup_survives_unreachable_source(self, mock_db, mock_get_stream, frame_queue, stop_event):
        """StreamLoader.setup() should not raise when video source is unreachable."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source: rtsp://bad")
        mock_db.return_value.__enter__ = Mock(return_value=Mock())
        mock_db.return_value.__exit__ = Mock(return_value=False)

        loader = self._make_loader(frame_queue, stop_event, None)

        with patch("app.workers.stream_loading.SourceService") as mock_source_service:
            mock_source_service.return_value.get_active_source.return_value = VideoFileSourceConfig(
                source_type=SourceType.VIDEO_FILE,
                id=uuid4(),
                name="Test",
                config_data=VideoFileConfig(video_path="fake.mp4"),
            )
            # setup() calls _load_source() which calls _reset_stream(); should not raise
            loader.setup()

        assert mock_get_stream.called
        assert loader._video_stream is None
        # run_loop should handle None stream gracefully
        stop_event.set()
        loader.run_loop()
