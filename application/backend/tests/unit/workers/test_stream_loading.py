# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import time
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.models import SourceType, VideoFileSourceConfig
from app.models.source import SourceStatus, SourceStatusCode, VideoFileConfig
from app.stream.stream_data import StreamData
from app.workers import StreamLoader
from app.workers.shm_status import STATUS_SHM_SIZE, read_status


class _FakeVideoStream:
    """Minimal picklable stand-in for VideoStream."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def get_data(self) -> StreamData:
        return StreamData(frame_data=self._frame, timestamp=time.time(), source_metadata={})

    def is_real_time(self) -> bool:
        return True

    def is_finished(self) -> bool:
        return False

    def release(self) -> None:
        pass


class _ExhaustibleVideoStream:
    """Stand-in for a finite (non-looping) VideoStream that finishes after a fixed number of frames."""

    def __init__(self, frame: np.ndarray, num_frames: int) -> None:
        self._frame = frame
        self._remaining = num_frames
        self._exhausted = False
        self.released = False
        self.finished_calls = 0

    def get_data(self) -> StreamData | None:
        if self._remaining > 0:
            self._remaining -= 1
            return StreamData(frame_data=self._frame, timestamp=time.time(), source_metadata={})
        self._exhausted = True
        return None

    def is_real_time(self) -> bool:
        return False

    def is_finished(self) -> bool:
        self.finished_calls += 1
        return self._exhausted

    def release(self) -> None:
        self.released = True


class _TestableStreamLoader(StreamLoader):
    def __init__(
        self, frame_queue, status_shm_name, status_shm_lock, stop_event, source_changed_condition, fake_video_stream
    ):
        super().__init__(
            frame_queue=frame_queue,
            status_shm_name=status_shm_name,
            status_shm_lock=status_shm_lock,
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
def status_shm():
    """Real backing shared-memory block standing in for the scheduler-owned status shm.

    A real SharedMemory (rather than a Mock) is required because the loader runs in a
    spawned process and `setup()` opens the block by name; the name and lock must also be
    picklable across the spawn boundary, which Mock objects are not.
    """
    shm = SharedMemory(create=True, size=STATUS_SHM_SIZE)
    try:
        yield shm
    finally:
        shm.close()
        shm.unlink()


@pytest.fixture
def status_shm_lock(mp_ctx):
    return mp_ctx.Lock()


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_video_stream(sample_frame) -> _FakeVideoStream:
    return _FakeVideoStream(frame=sample_frame)


class TestStreamLoader:
    def _start_loader(
        self,
        request,
        frame_queue,
        status_shm_name,
        status_shm_lock,
        stop_event,
        source_changed_condition,
        fake_video_stream,
    ):
        """Helper that starts a _TestableStreamLoader and registers cleanup."""
        process = _TestableStreamLoader(
            frame_queue, status_shm_name, status_shm_lock, stop_event, source_changed_condition, fake_video_stream
        )
        process.start()

        def cleanup():
            stop_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        request.addfinalizer(cleanup)
        return process

    def test_queue_full(
        self,
        request,
        frame_queue,
        status_shm,
        status_shm_lock,
        stop_event,
        source_changed_condition,
        fake_video_stream,
        sample_frame,
    ):
        """Test that for real-time sources the loader drops stale frames and replaces with fresh ones."""
        sentinel = np.zeros_like(sample_frame)
        data1 = StreamData(frame_data=sentinel.copy(), timestamp=time.time(), source_metadata={"src": "pre1"})
        data2 = StreamData(frame_data=sentinel.copy(), timestamp=time.time(), source_metadata={"src": "pre2"})
        frame_queue.put(data1)
        frame_queue.put(data2)

        process = self._start_loader(
            request,
            frame_queue,
            status_shm.name,
            status_shm_lock,
            stop_event,
            source_changed_condition,
            fake_video_stream,
        )

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

    def test_queue_empty(
        self, request, frame_queue, status_shm, status_shm_lock, stop_event, source_changed_condition, fake_video_stream
    ):
        """Test that stream frames are acquired when queue is empty."""
        process = self._start_loader(
            request,
            frame_queue,
            status_shm.name,
            status_shm_lock,
            stop_event,
            source_changed_condition,
            fake_video_stream,
        )

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

    def _make_loader(self, frame_queue, status_shm_name, status_shm_lock, stop_event, source_changed_condition):
        return StreamLoader(
            frame_queue=frame_queue,
            status_shm_name=status_shm_name,
            status_shm_lock=status_shm_lock,
            stop_event=stop_event,
            source_changed_condition=source_changed_condition,
            logger_=Mock(),
        )

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_reset_stream_handles_runtime_error(
        self, mock_get_stream, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """_reset_stream should catch RuntimeError and set _video_stream to None."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source: rtsp://bad")
        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event, None)
        loader._reset_stream()
        assert loader._video_stream is None

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_reset_stream_releases_existing_stream_on_error(
        self, mock_get_stream, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """_reset_stream should release the old stream even when the new one fails."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source")
        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event, None)
        old_stream = Mock()
        loader._video_stream = old_stream
        loader._reset_stream()
        old_stream.release.assert_called_once()
        assert loader._video_stream is None

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    @patch("app.workers.stream_loading.get_db_session")
    def test_reload_source_loop_survives_error(
        self, mock_db, mock_get_stream, mp_ctx, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """_reload_source_loop should not crash when _load_source raises."""
        mock_db.return_value.__enter__ = Mock(return_value=Mock())
        mock_db.return_value.__exit__ = Mock(return_value=False)
        mock_get_stream.side_effect = RuntimeError("unreachable")

        condition = mp_ctx.Condition()
        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event, condition)

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
    def test_setup_survives_unreachable_source(
        self, mock_db, mock_get_stream, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """StreamLoader.setup() should not raise when video source is unreachable."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source: rtsp://bad")
        mock_db.return_value.__enter__ = Mock(return_value=Mock())
        mock_db.return_value.__exit__ = Mock(return_value=False)

        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event, None)

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


def _video_file_source() -> VideoFileSourceConfig:
    return VideoFileSourceConfig(
        source_type=SourceType.VIDEO_FILE,
        id=uuid4(),
        name="Test Video File Source",
        config_data=VideoFileConfig(video_path="fake_path.mp4"),
    )


class TestStreamLoaderFinishedStream:
    """Tests for stopping a finite stream once it has been fully consumed (e.g. non-looping video)."""

    def _make_loader(self, frame_queue, status_shm_name, status_shm_lock, stop_event, source_changed_condition=None):
        return StreamLoader(
            frame_queue=frame_queue,
            status_shm_name=status_shm_name,
            status_shm_lock=status_shm_lock,
            stop_event=stop_event,
            source_changed_condition=source_changed_condition,
            logger_=Mock(),
        )

    def test_run_loop_releases_and_clears_finished_stream(
        self, mp_manager, status_shm, status_shm_lock, stop_event, sample_frame
    ):
        """When the stream finishes, run_loop releases it and clears it instead of polling forever."""
        frame_queue = mp_manager.Queue(maxsize=10)
        fake = _ExhaustibleVideoStream(frame=sample_frame, num_frames=3)
        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event)
        loader._source = _video_file_source()
        loader._video_stream = fake  # pyrefly: ignore[bad-assignment]

        thread = Thread(target=loader.run_loop, daemon=True)
        thread.start()
        try:
            # Wait until the finished stream has been released by the loader.
            deadline = time.monotonic() + 5
            while not fake.released and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            stop_event.set()
            thread.join(timeout=3)

        assert fake.released is True, "Finished stream should be released"
        assert loader._video_stream is None, "Finished stream should be cleared so polling stops"
        assert fake.finished_calls > 0, "Loader should have checked is_finished() when get_data() returned None"
        assert not thread.is_alive()

        # Exactly the produced frames were enqueued (no None frames, no infinite retries).
        drained = []
        while not frame_queue.empty():
            try:
                drained.append(frame_queue.get_nowait())
            except queue.Empty:
                break
        assert len(drained) == 3

    def test_run_loop_keeps_polling_when_not_finished(
        self, frame_queue, status_shm, status_shm_lock, stop_event, sample_frame
    ):
        """A transient None (stream not finished) must not stop the stream; it keeps the stream alive."""
        stream = Mock()
        stream.get_data.return_value = None
        stream.is_finished.return_value = False
        stream.is_real_time.return_value = False

        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event)
        loader._source = _video_file_source()
        loader._video_stream = stream

        thread = Thread(target=loader.run_loop, daemon=True)
        thread.start()
        try:
            # Give the loop time to run through several polling iterations.
            deadline = time.monotonic() + 1
            while stream.get_data.call_count < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            stop_event.set()
            thread.join(timeout=3)

        stream.release.assert_not_called()
        assert loader._video_stream is stream, "A non-finished stream must not be cleared"

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_finished_stream_can_be_re_enabled_via_source_reload(
        self, mock_get_stream, frame_queue, status_shm, status_shm_lock, stop_event, sample_frame
    ):
        """After a non-looping stream finished and was stopped (cleared to None), reloading the source
        (e.g. with looping enabled) re-creates a working stream that produces frames again."""
        loader = self._make_loader(frame_queue, status_shm.name, status_shm_lock, stop_event)
        # Simulate the state left behind after a finished stream was stopped.
        loader._source = _video_file_source()
        loader._video_stream = None

        # The reloaded source yields a fresh (looping) stream that never finishes.
        looping_stream = _FakeVideoStream(frame=sample_frame)
        mock_get_stream.return_value = looping_stream

        loader._reset_stream()

        assert loader._video_stream is looping_stream, "Reloading the source should re-create the stream"
        assert loader._video_stream.get_data() is not None, "Re-enabled stream should produce frames again"
        assert loader._video_stream.is_finished() is False


class TestStreamLoaderStatusReporting:
    """Tests that the loader reports OK and ERROR source statuses to shared memory."""

    def _make_loader(self, frame_queue, status_shm, status_shm_lock, stop_event):
        loader = StreamLoader(
            frame_queue=frame_queue,
            status_shm_name=status_shm.name,
            status_shm_lock=status_shm_lock,
            stop_event=stop_event,
            source_changed_condition=None,
            logger_=Mock(),
        )
        # setup() would open the shared-memory block by name; assign the real block directly instead.
        loader._status_shm = status_shm
        loader._source = _video_file_source()
        return loader

    def test_reports_ok_status_after_successful_frame(
        self, frame_queue, status_shm, status_shm_lock, stop_event, sample_frame
    ):
        """A successfully acquired and enqueued frame reports an OK status to shared memory."""
        stream = Mock()
        stream.get_data.return_value = StreamData(frame_data=sample_frame, timestamp=time.time(), source_metadata={})
        stream.is_real_time.return_value = False
        stream.is_finished.return_value = False

        loader = self._make_loader(frame_queue, status_shm, status_shm_lock, stop_event)
        loader._video_stream = stream

        thread = Thread(target=loader.run_loop, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while frame_queue.empty() and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            stop_event.set()
            thread.join(timeout=3)

        status = read_status(SourceStatus, status_shm, status_shm_lock)
        assert status is not None
        assert status.code == SourceStatusCode.OK
        assert status.source_id == loader._source.id

    def test_reports_error_status_on_frame_acquisition_failure(
        self, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """A failure while acquiring a frame reports an ERROR status to shared memory."""
        stream = Mock()
        stream.get_data.side_effect = RuntimeError("boom")
        stream.is_real_time.return_value = False
        stream.is_finished.return_value = False

        loader = self._make_loader(frame_queue, status_shm, status_shm_lock, stop_event)
        loader._video_stream = stream

        thread = Thread(target=loader.run_loop, daemon=True)
        thread.start()
        status = None
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = read_status(SourceStatus, status_shm, status_shm_lock)
                if status is not None and status.code == SourceStatusCode.ERROR:
                    break
                time.sleep(0.02)
        finally:
            stop_event.set()
            thread.join(timeout=3)

        assert status is not None
        assert status.code == SourceStatusCode.ERROR
        assert status.message == "Error acquiring frame"
        assert status.source_id == loader._source.id

    @patch("app.workers.stream_loading.VideoStreamService.get_video_stream")
    def test_reset_stream_reports_error_status_on_open_failure(
        self, mock_get_stream, frame_queue, status_shm, status_shm_lock, stop_event
    ):
        """When opening the video stream fails, _reset_stream reports an ERROR status."""
        mock_get_stream.side_effect = RuntimeError("Could not open video source")
        loader = self._make_loader(frame_queue, status_shm, status_shm_lock, stop_event)

        loader._reset_stream()

        status = read_status(SourceStatus, status_shm, status_shm_lock)
        assert status is not None
        assert status.code == SourceStatusCode.ERROR
        assert status.message is not None
        assert loader._source.name in status.message
