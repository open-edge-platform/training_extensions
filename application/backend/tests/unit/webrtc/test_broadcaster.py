# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from app.webrtc.broadcaster import FrameBroadcaster


class TestFrameBroadcaster:
    def test_register_creates_queue(self):
        """Test that registering a consumer creates a queue."""
        broadcaster = FrameBroadcaster[str]()
        queue = broadcaster.register("consumer1")

        assert queue is not None
        assert broadcaster.is_registered("consumer1")
        assert broadcaster.consumer_count() == 1

    def test_register_seeds_latest_frame(self):
        """Test that newly registered consumers receive the latest frame."""
        broadcaster = FrameBroadcaster[str]()

        # Broadcast a frame before registration
        broadcaster.broadcast("frame1")

        # Register a new consumer
        queue = broadcaster.register("consumer1")

        # The queue should contain the latest frame
        assert not queue.empty()
        frame = queue.get_nowait()
        assert frame == "frame1"

    def test_register_duplicate_returns_existing_queue(self):
        """Test that registering the same consumer twice returns the existing queue."""
        broadcaster = FrameBroadcaster[str]()

        queue1 = broadcaster.register("consumer1")
        queue2 = broadcaster.register("consumer1")

        # Should return the same queue
        assert queue1 is queue2
        assert broadcaster.consumer_count() == 1

    def test_unregister_removes_consumer(self):
        """Test that unregistering removes a consumer."""
        broadcaster = FrameBroadcaster[str]()
        broadcaster.register("consumer1")

        assert broadcaster.is_registered("consumer1")

        broadcaster.unregister("consumer1")

        assert not broadcaster.is_registered("consumer1")
        assert broadcaster.consumer_count() == 0

    def test_unregister_nonexistent_consumer_doesnt_raise(self):
        """Test that unregistering a non-existent consumer doesn't raise an error."""
        broadcaster = FrameBroadcaster[str]()

        # Should not raise an exception
        broadcaster.unregister("nonexistent")

    def test_broadcast_sends_to_all_consumers(self):
        """Test that broadcasting sends frames to all registered consumers."""
        broadcaster = FrameBroadcaster[str]()

        queue1 = broadcaster.register("consumer1")
        queue2 = broadcaster.register("consumer2")
        queue3 = broadcaster.register("consumer3")

        # Clear initial frames from registration
        while not queue1.empty():
            queue1.get_nowait()
        while not queue2.empty():
            queue2.get_nowait()
        while not queue3.empty():
            queue3.get_nowait()

        broadcaster.broadcast("test_frame")

        # All queues should have the frame
        assert queue1.get_nowait() == "test_frame"
        assert queue2.get_nowait() == "test_frame"
        assert queue3.get_nowait() == "test_frame"

    def test_broadcast_updates_latest_frame(self):
        """Test that broadcasting updates the latest frame."""
        broadcaster = FrameBroadcaster[str]()

        assert broadcaster.latest_frame is None

        broadcaster.broadcast("frame1")
        assert broadcaster.latest_frame == "frame1"

        broadcaster.broadcast("frame2")
        assert broadcaster.latest_frame == "frame2"

    def test_broadcast_drops_oldest_when_queue_full(self):
        """Test that broadcasting drops the oldest frame when a queue is full."""
        broadcaster = FrameBroadcaster[int]()
        queue = broadcaster.register("consumer1")

        # Clear initial state
        while not queue.empty():
            queue.get_nowait()

        # Fill the queue to capacity (maxsize=5)
        for i in range(5):
            broadcaster.broadcast(i)

        # Queue should be full
        assert queue.qsize() == 5

        # Broadcast another frame - should drop oldest
        broadcaster.broadcast(99)

        # Queue should still have 5 items
        assert queue.qsize() == 5

        # First frame should be 1 (0 was dropped)
        assert queue.get_nowait() == 1

        # Verify remaining frames
        assert queue.get_nowait() == 2
        assert queue.get_nowait() == 3
        assert queue.get_nowait() == 4
        assert queue.get_nowait() == 99

    def test_clear_drains_all_queues(self):
        """Test that clear drains all consumer queues."""
        broadcaster = FrameBroadcaster[str]()

        queue1 = broadcaster.register("consumer1")
        queue2 = broadcaster.register("consumer2")

        # Broadcast some frames
        broadcaster.broadcast("frame1")
        broadcaster.broadcast("frame2")
        broadcaster.broadcast("frame3")

        # Queues should have frames
        assert not queue1.empty()
        assert not queue2.empty()

        # Clear all queues
        broadcaster.clear()

        # Queues should be empty
        assert queue1.empty()
        assert queue2.empty()

        # Latest frame should be reset
        assert broadcaster.latest_frame is None

    def test_clear_keeps_consumers_registered(self):
        """Test that clear drains queues but keeps consumers registered."""
        broadcaster = FrameBroadcaster[str]()

        broadcaster.register("consumer1")
        broadcaster.register("consumer2")
        broadcaster.broadcast("frame1")

        assert broadcaster.consumer_count() == 2

        broadcaster.clear()

        # Consumers should still be registered
        assert broadcaster.consumer_count() == 2
        assert broadcaster.is_registered("consumer1")
        assert broadcaster.is_registered("consumer2")

    def test_consumer_count(self):
        """Test consumer_count returns correct number."""
        broadcaster = FrameBroadcaster[str]()

        assert broadcaster.consumer_count() == 0

        broadcaster.register("consumer1")
        assert broadcaster.consumer_count() == 1

        broadcaster.register("consumer2")
        assert broadcaster.consumer_count() == 2

        broadcaster.unregister("consumer1")
        assert broadcaster.consumer_count() == 1

        broadcaster.unregister("consumer2")
        assert broadcaster.consumer_count() == 0

    def test_is_registered(self):
        """Test is_registered correctly identifies registered consumers."""
        broadcaster = FrameBroadcaster[str]()

        assert not broadcaster.is_registered("consumer1")

        broadcaster.register("consumer1")
        assert broadcaster.is_registered("consumer1")
        assert not broadcaster.is_registered("consumer2")

        broadcaster.register("consumer2")
        assert broadcaster.is_registered("consumer1")
        assert broadcaster.is_registered("consumer2")

        broadcaster.unregister("consumer1")
        assert not broadcaster.is_registered("consumer1")
        assert broadcaster.is_registered("consumer2")
