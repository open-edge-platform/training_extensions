# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Callable, Sequence
from threading import RLock
from typing import TypeVar

from loguru import logger

E = TypeVar("E")


class BaseEventBus[E]:
    def __init__(self) -> None:
        self._event_handlers: dict[E, list[Callable]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_types: Sequence[E], handler: Callable) -> None:
        with self._lock:
            for event_type in event_types:
                self._event_handlers[event_type].append(handler)
                logger.debug(f"registered event handler for event '{event_type}'")

    def emit_event(self, event_type: E) -> None:
        logger.debug(f"Emitting event '{event_type}' to {len(self._event_handlers[event_type])} handlers")
        with self._lock:
            for handler in self._event_handlers[event_type]:
                handler()
