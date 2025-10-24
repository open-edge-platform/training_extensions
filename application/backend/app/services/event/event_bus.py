# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from multiprocessing.synchronize import Condition


class Listener:
    """
    Events listener, gets callback notifications upon certain events, i.e. source/sink change, pipeline update.
    """

    def on_source_changed(self) -> None:
        """
        Callback method for source changes.
        """

    def on_sink_changed(self) -> None:
        """
        Callback method for sink changes.
        """

    def on_pipeline_dataset_collection_policies_changed(self) -> None:
        """
        Callback method for dataset collection policies changes.
        """

    def on_pipeline_status_changed(self) -> None:
        """
        Callback method for pipeline status changes.
        """


class EventBus:
    def __init__(
        self, source_changed_condition: Condition | None = None, sink_changed_condition: Condition | None = None
    ) -> None:
        self._listeners: list[Listener] = []
        self._source_changed_condition = source_changed_condition
        self._sink_changed_condition = sink_changed_condition

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def source_changed(self) -> None:
        for listener in self._listeners:
            listener.on_source_changed()
        if self._source_changed_condition:
            with self._source_changed_condition:
                self._source_changed_condition.notify_all()

    def sink_changed(self) -> None:
        for listener in self._listeners:
            listener.on_sink_changed()
        if self._sink_changed_condition:
            with self._sink_changed_condition:
                self._sink_changed_condition.notify_all()

    def pipeline_dataset_collection_policies_changed(self) -> None:
        for listener in self._listeners:
            listener.on_pipeline_dataset_collection_policies_changed()

    def pipeline_status_changed(self) -> None:
        for listener in self._listeners:
            listener.on_pipeline_status_changed()
        if self._source_changed_condition:
            with self._source_changed_condition:
                self._source_changed_condition.notify_all()
        if self._sink_changed_condition:
            with self._sink_changed_condition:
                self._sink_changed_condition.notify_all()
