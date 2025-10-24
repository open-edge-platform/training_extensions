# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest

from app.services.event.event_bus import EventBus, Listener


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

    def test_subscribe(self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]) -> None:
        """Test subscription"""
        listener = MagicMock(spec=Listener)
        event_bus = fxt_event_bus(None, None)

        event_bus.subscribe(listener)

        assert event_bus._listeners == [listener]

    def test_source_changed(self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]) -> None:
        """Test source changed"""
        listener = MagicMock(spec=Listener)
        source_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(source_changed_condition, None)
        event_bus.subscribe(listener)

        event_bus.source_changed()

        listener.on_source_changed.assert_called_once_with()
        with source_changed_condition:
            notified = source_changed_condition.acquire()
        assert notified

    def test_sink_changed(self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]) -> None:
        """Test sink changed"""
        listener = MagicMock(spec=Listener)
        sink_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(None, sink_changed_condition)
        event_bus.subscribe(listener)

        event_bus.sink_changed()

        listener.on_sink_changed.assert_called_once_with()
        with sink_changed_condition:
            notified = sink_changed_condition.acquire()
        assert notified

    def test_pipeline_dataset_collection_policies_changed(
        self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]
    ) -> None:
        """Test pipeline dataset collection policies changed"""
        listener = MagicMock(spec=Listener)
        event_bus = fxt_event_bus(None, None)
        event_bus.subscribe(listener)

        event_bus.pipeline_dataset_collection_policies_changed()

        listener.on_pipeline_dataset_collection_policies_changed.assert_called_once_with()

    def test_pipeline_status_changed(
        self, fxt_event_bus: Callable[[Condition | None, Condition | None], EventBus]
    ) -> None:
        """Test pipeline status changed"""
        listener = MagicMock(spec=Listener)
        source_changed_condition = mp.Condition()
        sink_changed_condition = mp.Condition()
        event_bus = fxt_event_bus(source_changed_condition, sink_changed_condition)
        event_bus.subscribe(listener)

        event_bus.pipeline_status_changed()

        listener.on_pipeline_status_changed.assert_called_once_with()
        with source_changed_condition:
            notified = source_changed_condition.acquire()
        assert notified
        with sink_changed_condition:
            notified = sink_changed_condition.acquire()
        assert notified
