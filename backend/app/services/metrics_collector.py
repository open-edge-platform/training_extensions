# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections import deque
from datetime import UTC, datetime
from threading import Lock
from typing import NamedTuple
from uuid import UUID

from app.utils import Singleton

logger = logging.getLogger(__name__)


class LatencyMeasurement(NamedTuple):
    """Individual latency measurement"""

    model_id: UUID
    latency_ms: float
    timestamp: datetime


class MetricsCollector(metaclass=Singleton):
    """Thread-safe in-memory metrics collector for model latency data"""

    def __init__(self, max_age_seconds: int = 60):
        self._max_age_seconds = max_age_seconds
        self._measurements: deque[LatencyMeasurement] = deque()
        self._lock = Lock()

    def update_max_age(self, max_age_seconds: int) -> None:
        """Update the maximum age for stored measurements"""
        with self._lock:
            self._max_age_seconds = max_age_seconds
            self._cleanup_old_measurements()

    @staticmethod
    def record_inference_start() -> float:
        """Record the start of an inference operation and return the start timestamp"""
        return time.perf_counter()

    def record_inference_end(self, model_id: UUID, start_time: float) -> None:
        """Record the end of an inference operation and store the latency measurement"""
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0  # Convert to milliseconds

        measurement = LatencyMeasurement(model_id=model_id, latency_ms=latency_ms, timestamp=datetime.now(UTC))

        with self._lock:
            logger.debug(f"Latency measurement recorded for model {model_id}: {latency_ms:.2f} ms")
            self._measurements.append(measurement)
            self._cleanup_old_measurements()

    def get_latency_measurements(self, model_id: UUID, time_window: int = 60) -> list[float]:
        """Get latency measurements for a specific model within the time window"""
        cutoff_time = datetime.now(UTC).timestamp() - time_window

        with self._lock:
            self._cleanup_old_measurements()
            return [
                m.latency_ms
                for m in self._measurements
                if m.model_id == model_id and m.timestamp.timestamp() >= cutoff_time
            ]

    def _cleanup_old_measurements(self) -> None:
        """Remove measurements older than max_age_seconds (called with lock held)"""
        cutoff_time = datetime.now(UTC).timestamp() - self._max_age_seconds

        while self._measurements and self._measurements[0].timestamp.timestamp() < cutoff_time:
            self._measurements.popleft()

    def reset(self) -> None:
        """Reset the metrics collector"""
        with self._lock:
            self._max_age_seconds = 60
            self._measurements.clear()
