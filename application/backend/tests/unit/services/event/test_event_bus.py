# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest

from app.services.event.event_bus import EventBus, EventType


@pytest.fixture
def fxt_event_bus() -> Callable[[Condition | None, Condition | None], EventBus]:
    def _create_bus(source_changed_condition: Condition | None, sink_changed_condition: Condition | None) -> EventBus:
        return EventBus(
            source_changed_condition=source_changed_condition,
            sink_changed_condition=sink_changed_condition,
        )

    return _create_bus


class TestEventBus:
    """Unit tests for TestEventBus."""

    @pytest.mark.parametrize("event_type", EventType)
    def test_subscribe(
        self, event_type: EventType, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]
    ) -> None:
        """Test subscription"""
        handler = MagicMock(spec=Callable)
        event_bus = fxt_event_bus(None, None)

        event_bus.subscribe(event_types=[event_type], handler=handler)

        assert event_bus._event_handlers == {event_type: [handler]}

    def test_source_changed(self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]) -> None:
        """Test source changed"""
        handler = MagicMock(spec=Callable)
        source_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(source_changed_condition, None)
        event_bus.subscribe(event_types=[EventType.SOURCE_CHANGED], handler=handler)

        event_bus.emit_event(EventType.SOURCE_CHANGED)

        handler.assert_called_once_with()
        with source_changed_condition:
            notified = source_changed_condition.acquire()
        assert notified

    def test_sink_changed(self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]) -> None:
        """Test sink changed"""
        handler = MagicMock(spec=Callable)
        sink_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(None, sink_changed_condition)
        event_bus.subscribe(event_types=[EventType.SINK_CHANGED], handler=handler)

        event_bus.emit_event(EventType.SINK_CHANGED)

        handler.assert_called_once_with()
        with sink_changed_condition:
            notified = sink_changed_condition.acquire()
        assert notified

    def test_pipeline_dataset_collection_policies_changed(
        self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]
    ) -> None:
        """Test pipeline dataset collection policies changed"""
        handler = MagicMock(spec=Callable)
        event_bus = fxt_event_bus(None, None)
        event_bus.subscribe(event_types=[EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED], handler=handler)

        event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)

        handler.assert_called_once_with()

    def test_pipeline_status_changed(
        self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]
    ) -> None:
        """Test pipeline status changed"""
        handler = MagicMock(spec=Callable)
        source_changed_condition = mp.Condition()
        sink_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(source_changed_condition, sink_changed_condition)
        event_bus.subscribe(event_types=[EventType.PIPELINE_STATUS_CHANGED], handler=handler)

        event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)

        handler.assert_called_once_with()
        with source_changed_condition:
            notified = source_changed_condition.acquire()
        assert notified
        with sink_changed_condition:
            notified = sink_changed_condition.acquire()
        assert notified
