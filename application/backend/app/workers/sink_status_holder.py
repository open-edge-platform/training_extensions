# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Thread-safe holder for the latest SinkStatus."""

import threading

from app.models.sink import SinkStatus


class SinkStatusHolder:
    """A thread-safe container for the latest SinkStatus value."""

    def __init__(self) -> None:
        self._status: SinkStatus | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> SinkStatus | None:
        with self._lock:
            return self._status

    @status.setter
    def status(self, value: SinkStatus) -> None:
        with self._lock:
            self._status = value
