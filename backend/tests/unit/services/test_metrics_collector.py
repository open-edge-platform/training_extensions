# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from app.services.metrics_collector import LatencyMeasurement, MetricsCollector


class TestMetricsCollector:
    """Test cases for MetricsCollector"""

    collector: MetricsCollector = MetricsCollector()

    def reset_collector(self):
        """Reset the singleton instance for isolated tests"""
        self.collector.reset()

    def test_singleton_behavior(self):
        """Test that MetricsCollector follows singleton pattern"""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()

        assert collector1 is collector2

    def test_record_inference_start(self):
        """Test recording inference start time"""
        start_time = MetricsCollector.record_inference_start()

        assert isinstance(start_time, float)
        assert start_time > 0

    def test_record_inference_end(self):
        """Test recording inference end and latency calculation"""
        model_id = uuid4()

        start_time = time.perf_counter()
        time.sleep(0.01)  # Small delay to ensure measurable latency

        self.collector.record_inference_end(model_id=model_id, start_time=start_time)

        measurements = self.collector.get_latency_measurements(model_id=model_id)
        assert len(measurements) == 1
        assert measurements[0] >= 0.01  # Should have some latency

    def test_get_latency_measurements_with_time_window(self):
        """Test getting latency measurements within specific time window"""
        model_id = uuid4()

        # Record multiple measurements
        for _ in range(3):
            start_time = time.perf_counter()
            time.sleep(0.01)
            self.collector.record_inference_end(model_id=model_id, start_time=start_time)

        # Get measurements from last 60 seconds
        measurements = self.collector.get_latency_measurements(model_id=model_id, time_window=60)
        assert len(measurements) == 3

        # Get measurements from last 0 seconds (should be empty)
        measurements = self.collector.get_latency_measurements(model_id=model_id, time_window=0)
        assert len(measurements) == 0

    def test_get_latency_measurements_different_models(self):
        """Test that measurements are isolated by model ID"""
        model_id_1 = uuid4()
        model_id_2 = uuid4()

        # Record measurements for both models
        start_time = time.perf_counter()
        self.collector.record_inference_end(model_id=model_id_1, start_time=start_time)
        self.collector.record_inference_end(model_id=model_id_2, start_time=start_time)

        time.sleep(0.01)
        start_time = time.perf_counter()
        self.collector.record_inference_end(model_id=model_id_2, start_time=start_time)

        # Each model should only see its own measurements
        measurements_1 = self.collector.get_latency_measurements(model_id=model_id_1)
        measurements_2 = self.collector.get_latency_measurements(model_id=model_id_2)

        assert len(measurements_1) == 1
        assert len(measurements_2) == 2

    def test_cleanup_old_measurements(self):
        """Test that old measurements are cleaned up automatically"""
        model_id = uuid4()

        # Create old measurement by mocking timestamp greater than max age of 60 seconds
        old_timestamp = datetime.now(UTC) - timedelta(seconds=61)
        old_measurement = LatencyMeasurement(model_id=model_id, latency_ms=10.0, timestamp=old_timestamp)

        with self.collector._lock:
            self.collector._measurements.append(old_measurement)

        # Trigger cleanup by getting measurements
        measurements = self.collector.get_latency_measurements(model_id=model_id)

        # Old measurement should be cleaned up
        assert len(measurements) == 0

    def test_latency_measurement_timestamp(self):
        """Test that measurements are recorded with correct timestamps"""
        self.reset_collector()

        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        model_id = uuid4()
        start_time = self.collector.record_inference_start()
        with patch("app.services.metrics_collector.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            self.collector.record_inference_end(model_id, start_time)

        # Check that measurement was recorded with mocked timestamp
        with self.collector._lock:
            assert len(self.collector._measurements) == 1
            assert self.collector._measurements[0].timestamp == fixed_time

    def test_empty_measurements(self):
        """Test behavior when no measurements exist"""
        model_id = uuid4()

        measurements = self.collector.get_latency_measurements(model_id=model_id)
        assert measurements == []

    def test_latency_calculation_accuracy(self):
        """Test that latency calculation is reasonably accurate"""
        model_id = uuid4()

        start_time = time.perf_counter()
        time.sleep(0.01)  # 10ms
        self.collector.record_inference_end(model_id=model_id, start_time=start_time)
        measurements = self.collector.get_latency_measurements(model_id=model_id)
        latency_ms = measurements[0]

        # Should be approximately 10ms (with some tolerance for timing precision)
        assert 10.0 <= latency_ms <= 11.0

    def test_max_age_initialization(self):
        """Test that max_age_seconds is properly initialized and updated"""
        self.reset_collector()
        assert self.collector._max_age_seconds == 60

        self.collector.update_max_age(120)
        assert self.collector._max_age_seconds == 120

        # Test default value
        collector2 = MetricsCollector()
        assert collector2._max_age_seconds == 120  # Should be same instance due to singleton
