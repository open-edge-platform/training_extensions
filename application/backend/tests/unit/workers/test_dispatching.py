# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

from app.models import DisconnectedSinkConfig, SinkType
from app.services.dispatchers import DispatchError
from app.services.event.event_bus import EventBus, EventType
from app.stream.stream_data import InferenceData, StreamData
from app.webrtc import FrameBroadcaster
from app.workers.dispatching import DispatchingWorker


def _make_stream_data() -> StreamData:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return StreamData(
        frame_data=frame,
        timestamp=0.0,
        source_metadata={},
        inference_data=InferenceData(
            prediction=MagicMock(),
            visualized_prediction=frame.copy(),
            model_id=uuid.uuid4(),
        ),
    )


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

    def test_dispatch_error_does_not_stop_loop(self):
        """A DispatchError from a dispatcher is caught; the loop continues and WebRTC broadcast still happens."""
        import queue

        event_bus = EventBus()
        broadcaster = FrameBroadcaster[np.ndarray]()

        failing_dispatcher = MagicMock()
        failing_dispatcher.dispatch.side_effect = DispatchError()

        pred_queue: queue.Queue = queue.Queue()
        pred_queue.put(_make_stream_data())
        pred_queue.put(_make_stream_data())

        with patch.object(
            DispatchingWorker,
            "_load_sink",
            return_value=(MagicMock(sink_type=SinkType.FOLDER), [failing_dispatcher]),
        ):
            worker = DispatchingWorker(
                event_bus=event_bus,
                pred_queue=pred_queue,  # type: ignore[arg-type]
                rtc_stream_broadcaster=broadcaster,
                stop_event=mp.Event(),
                data_collector=MagicMock(),
            )

        def stop_when_empty() -> bool:
            return pred_queue.empty()

        worker.should_stop = stop_when_empty
        worker.run_loop()

        assert failing_dispatcher.dispatch.call_count == 2
        assert broadcaster.latest_frame is not None
