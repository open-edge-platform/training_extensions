# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from multiprocessing.synchronize import Condition, Event

from .base import BaseEventBus


class EventType(StrEnum):
    SOURCE_CHANGED = "SOURCE_CHANGED"
    SINK_CHANGED = "SINK_CHANGED"
    MODEL_CHANGED = "MODEL_CHANGED"
    PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED = "PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED"
    PIPELINE_STATUS_CHANGED = "PIPELINE_STATUS_CHANGED"
    INFERENCE_DEVICE_CHANGED = "INFERENCE_DEVICE_CHANGED"


class EventBus(BaseEventBus[EventType]):
    def __init__(
        self,
        source_changed_condition: Condition | None = None,
        sink_changed_condition: Condition | None = None,
        model_reload_event: Event | None = None,
    ) -> None:
        super().__init__()
        self._source_changed_condition = source_changed_condition
        self._sink_changed_condition = sink_changed_condition
        self._model_reload_event = model_reload_event

    @property
    def source_changed_condition(self) -> Condition | None:
        return self._source_changed_condition

    @property
    def sink_changed_condition(self) -> Condition | None:
        return self._sink_changed_condition

    @property
    def model_reload_event(self) -> Event | None:
        return self._model_reload_event

    def _notify_all(self, condition: Condition | None) -> None:
        if not condition:
            return
        with condition:
            condition.notify_all()

    def _should_notify_source(self, event_type: EventType) -> bool:
        return event_type in (EventType.SOURCE_CHANGED, EventType.PIPELINE_STATUS_CHANGED)

    def _should_notify_sink(self, event_type: EventType) -> bool:
        return event_type in (EventType.SINK_CHANGED, EventType.PIPELINE_STATUS_CHANGED)

    def _should_notify_model(self, event_type: EventType) -> bool:
        return event_type in (EventType.MODEL_CHANGED, EventType.PIPELINE_STATUS_CHANGED)

    def emit_event(self, event_type: EventType) -> None:
        super().emit_event(event_type)

        if self._should_notify_source(event_type):
            self._notify_all(self._source_changed_condition)

        if self._should_notify_sink(event_type):
            self._notify_all(self._sink_changed_condition)

        if self._should_notify_model(event_type) and self._model_reload_event:
            self._model_reload_event.set()
