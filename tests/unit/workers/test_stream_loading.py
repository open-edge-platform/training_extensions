import multiprocessing as mp
import queue
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.entities.video_stream import VideoStream
from app.schemas.configuration import InputConfig
from app.workers import frame_acquisition_routine


@pytest.fixture
def mp_manager():
    """Multiprocessing manager fixture"""
    manager = mp.Manager()
    yield manager
    manager.shutdown()


@pytest.fixture
def frame_queue(mp_manager):
    """Frame queue fixture"""
    return mp_manager.Queue(maxsize=2)


@pytest.fixture
def stop_event():
    """Stop event fixture"""
    return mp.Event()


@pytest.fixture
def config_changed_condition():
    """Configuration changed condition fixture"""
    return mp.Condition()


@pytest.fixture
def mock_config():
    """Mock configuration fixture"""
    config = Mock(spec=InputConfig)
    config.source = "test_camera"
    config.resolution = (1920, 1080)
    return config


@pytest.fixture
def mock_video_stream():
    """Mock video stream fixture"""
    stream = Mock(spec=VideoStream)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    stream.get_frame.return_value = test_frame
    stream.is_real_time.return_value = True
    stream.release.return_value = None
    return stream


@pytest.fixture
def mock_services(mock_config, mock_video_stream):
    with (
        patch("app.workers.stream_loading.ConfigurationService") as mock_config_service,
        patch("app.workers.stream_loading.VideoStreamService") as mock_video_service,
    ):
        # Set up the mocks
        mock_config_instance = Mock()
        mock_config_instance.get_input_config.return_value = mock_config
        mock_config_service.return_value = mock_config_instance

        mock_video_service.get_video_stream.return_value = mock_video_stream

        yield {
            "config_service": mock_config_service,
            "video_service": mock_video_service,
            "config_instance": mock_config_instance,
            "video_stream": mock_video_stream,
        }


class TestFrameAcquisition:
    """Unit tests for the frame acquisition routine"""

    def test_queue_full(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that stream frames are not acquired when queue is full"""

        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_queue.put(frame1)
        frame_queue.put(frame2)

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, False)
        )
        process.start()

        # Let it run for a short time to attempt frame acquisition
        time.sleep(2)

        # Stop the process
        stop_event.set()
        process.join(timeout=1)

        queue_contents = []
        while not frame_queue.empty():
            try:
                queue_contents.append(frame_queue.get())
            except queue.Empty:
                break

        # Should still have our initial frames, proving new frames were ignored
        assert len(queue_contents) == 2
        assert np.array_equal(queue_contents, [frame1, frame2])
        assert not process.is_alive(), "Process should terminate cleanly"

    def test_queue_empty(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that stream frames are acquired when queue is empty"""

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, False)
        )
        process.start()

        time.sleep(2)

        stop_event.set()
        process.join(timeout=1)

        assert frame_queue.qsize() == 2
        assert not process.is_alive(), "Process should terminate cleanly"

    def test_cleanup(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that resources has been successfully released when process finished"""

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, True)
        )
        process.start()

        time.sleep(2)

        stop_event.set()
        process.join(timeout=1)

        assert frame_queue.empty()
        assert not process.is_alive(), "Process should terminate cleanly"
