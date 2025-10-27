# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import StrEnum
from multiprocessing.synchronize import Condition
from threading import RLock


class EventType(StrEnum):
    SOURCE_CHANGED = "SOURCE_CHANGED"
    SINK_CHANGED = "SINK_CHANGED"
    PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED = "PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED"
    PIPELINE_STATUS_CHANGED = "PIPELINE_STATUS_CHANGED"


class EventBus:
    def __init__(
        self, source_changed_condition: Condition | None = None, sink_changed_condition: Condition | None = None
    ) -> None:
        self._event_handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._source_changed_condition = source_changed_condition
        self._sink_changed_condition = sink_changed_condition

        self._lock = RLock()

    @property
    def source_changed_condition(self) -> Condition | None:
        return self._source_changed_condition

    @property
    def sink_changed_condition(self) -> Condition | None:
        return self._sink_changed_condition

    def subscribe(self, event_types: Sequence[EventType], handler: Callable) -> None:
        with self._lock:
            for event_type in event_types:
                self._event_handlers[event_type].append(handler)

    def emit_event(self, event_type: EventType) -> None:
        with self._lock:
            for handler in self._event_handlers[event_type]:
                handler()
        if (
            event_type in (EventType.SOURCE_CHANGED, EventType.PIPELINE_STATUS_CHANGED)
            and self._source_changed_condition
        ):
            with self._source_changed_condition:
                self._source_changed_condition.notify_all()
        if event_type in (EventType.SINK_CHANGED, EventType.PIPELINE_STATUS_CHANGED) and self._sink_changed_condition:
            with self._sink_changed_condition:
                self._sink_changed_condition.notify_all()
