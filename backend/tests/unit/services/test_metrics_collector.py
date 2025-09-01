# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from app.services.metrics_collector import MetricsCollector


class TestMetricsCollector:
    """Test cases for MetricsCollector"""

    def setup_method(self):
        """Setup a clean collector instance before each test"""
        self.collector = MetricsCollector()
        self.collector.reset()

    def teardown_method(self):
        """Clean up the collector after each test"""
        self.collector.reset()

    def test_singleton_behavior(self):
        """Test that MetricsCollector follows singleton pattern"""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()

        assert collector1 is collector2
        assert collector1._shm.name == collector2._shm.name

    def test_record_inference_start(self):
        """Test recording inference start time"""
        start_time = MetricsCollector.record_inference_start()

        assert isinstance(start_time, float)
        assert start_time > 0

    def test_record_inference_end(self):
        """Test recording inference end and latency calculation"""
        model_id = uuid4()

        start_time = time.perf_counter()
        time.sleep(0.01)  # Small delay of 10ms to ensure measurable latency

        self.collector.record_inference_end(model_id=model_id, start_time=start_time)

        measurements = self.collector.get_latency_measurements(model_id=model_id)
        assert len(measurements) == 1
        assert measurements[0] >= 10  # Should be at least 10ms

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

    def test_time_window_measurements(self):
        """Test that old measurements are filtered out by time window"""
        model_id = uuid4()

        # First add a current measurement
        start_time = time.perf_counter()
        self.collector.record_inference_end(model_id=model_id, start_time=start_time)

        # Then add an "old" measurement by mocking datetime
        with patch("app.services.metrics_collector.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now(UTC) - timedelta(seconds=90)
            self.collector.record_inference_end(model_id, start_time)

        # Only the first measurement should be returned with default 60s window
        measurements = self.collector.get_latency_measurements(model_id=model_id)
        assert len(measurements) == 1

        # Both should be returned with larger time window
        measurements = self.collector.get_latency_measurements(model_id=model_id, time_window=120)
        assert len(measurements) == 2

    def test_latency_measurement_timestamp(self):
        """Test that measurements are recorded with correct timestamps"""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        fixed_timestamp = fixed_time.timestamp()
        model_id = uuid4()
        start_time = self.collector.record_inference_start()

        with patch("app.services.metrics_collector.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            self.collector.record_inference_end(model_id, start_time)

        # Get all entries and check for our timestamp
        with self.collector._lock:
            # Check the array for our timestamp
            found_matching_record = False
            for i in range(self.collector.MAX_MEASUREMENTS):
                if (
                    self.collector._array[i]["model_id"] == str(model_id)
                    and abs(self.collector._array[i]["timestamp"] - fixed_timestamp) < 0.001
                ):
                    found_matching_record = True
                    break
            assert found_matching_record

    def test_empty_measurements(self):
        """Test behavior when no measurements exist"""
        self.collector.reset()
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
        assert 10.0 <= latency_ms <= 12.0, f"Latency was {latency_ms} ms"

    def test_max_age_initialization(self):
        """Test that max_age_seconds is properly initialized and updated"""
        assert self.collector._max_age_seconds == 60

        self.collector.update_max_age(120)
        assert self.collector._max_age_seconds == 120

        # Test default value
        collector2 = MetricsCollector()
        assert collector2._max_age_seconds == 120  # Should be same instance due to singleton

    def test_shared_memory_usage(self):
        """Test that the shared memory is properly used"""
        model_id = uuid4()
        start_time = time.perf_counter()
        self.collector.record_inference_end(model_id, start_time)

        # Create a new collector which should access the same shared memory
        collector2 = MetricsCollector()
        measurements = collector2.get_latency_measurements(model_id)

        assert len(measurements) == 1

    def test_circular_buffer(self):
        """Test that the circular buffer works correctly when exceeding MAX_MEASUREMENTS"""
        model_id = uuid4()

        # Fill more than MAX_MEASUREMENTS entries
        head_before = self.collector._head
        for i in range(self.collector.MAX_MEASUREMENTS + 10):
            self.collector.record_inference_end(model_id, time.perf_counter())

        # Check that the head has wrapped around correctly
        assert self.collector._head == head_before + self.collector.MAX_MEASUREMENTS + 10

        # Check that we can still retrieve measurements
        measurements = self.collector.get_latency_measurements(model_id)
        assert len(measurements) == self.collector.MAX_MEASUREMENTS
