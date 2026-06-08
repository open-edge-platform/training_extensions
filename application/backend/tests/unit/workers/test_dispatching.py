# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from unittest.mock import MagicMock, patch

import numpy as np

from app.models import DisconnectedSinkConfig
from app.services.event.event_bus import EventBus, EventType
from app.webrtc import FrameBroadcaster
from app.workers.dispatching import DispatchingWorker


def _make_worker(event_bus: EventBus, broadcaster: FrameBroadcaster[np.ndarray]) -> DispatchingWorker:
    with patch.object(DispatchingWorker, "_load_sink", return_value=(DisconnectedSinkConfig(), [])):
        return DispatchingWorker(
            event_bus=event_bus,
            pred_queue=mp.Queue(),
            rtc_stream_broadcaster=broadcaster,
            stop_event=mp.Event(),
            data_collector=MagicMock(),
        )


class TestDispatchingWorkerSourceChange:
    def test_source_changed_clears_broadcaster(self):
        """A SOURCE_CHANGED event drops the cached frame so new consumers don't see a stale frame."""
        event_bus = EventBus()
        broadcaster = FrameBroadcaster[np.ndarray]()
        _make_worker(event_bus, broadcaster)

        # A frame from the previous source is cached and would be seeded to new consumers.
        broadcaster.broadcast(np.zeros((2, 2, 3), dtype=np.uint8))
        assert broadcaster.latest_frame is not None

        event_bus.emit_event(EventType.SOURCE_CHANGED)

        # The cached frame is dropped, so a freshly connecting consumer gets nothing stale.
        assert broadcaster.latest_frame is None
        new_consumer_queue = broadcaster.register("new-consumer")
        assert new_consumer_queue.empty()

    def test_source_changed_drains_existing_consumer_queues(self):
        """Already-connected consumers also have their queued stale frames drained."""
        event_bus = EventBus()
        broadcaster = FrameBroadcaster[np.ndarray]()
        _make_worker(event_bus, broadcaster)

        consumer_queue = broadcaster.register("consumer")
        broadcaster.broadcast(np.zeros((2, 2, 3), dtype=np.uint8))
        assert not consumer_queue.empty()

        event_bus.emit_event(EventType.SOURCE_CHANGED)

        assert consumer_queue.empty()

    def test_sink_changed_does_not_clear_broadcaster(self):
        """A SINK_CHANGED event must not drop the cached WebRTC frame."""
        event_bus = EventBus()
        broadcaster = FrameBroadcaster[np.ndarray]()
        _make_worker(event_bus, broadcaster)

        broadcaster.broadcast(np.zeros((2, 2, 3), dtype=np.uint8))

        with patch.object(DispatchingWorker, "_load_sink", return_value=(DisconnectedSinkConfig(), [])):
            event_bus.emit_event(EventType.SINK_CHANGED)

        assert broadcaster.latest_frame is not None
