# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import UTC, datetime, timedelta
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.services.metrics_service import MAX_MEASUREMENTS, SHM_NAME, SIZE, MetricsService


@pytest.fixture(scope="module", autouse=True)
def fxt_metrics_mem():
    shm = SharedMemory(name=SHM_NAME, create=True, size=SIZE)
    yield
    shm.close()
    shm.unlink()


class TestMetricsCollector:
    """Test cases for MetricsService"""

    def test_record_inference_start(self):
        """Test recording inference start time"""
        start_time = MetricsService.record_inference_start()

        assert isinstance(start_time, float)
        assert start_time > 0

    def test_record_inference_end(self):
        """Test recording inference end and latency calculation"""
        collector = MetricsService()
        model_id = uuid4()

        start_time = time.perf_counter()
        time.sleep(0.01)  # Small delay of 10ms to ensure measurable latency

        collector.record_inference_end(model_id=model_id, start_time=start_time)

        measurements = collector.get_latency_measurements(model_id=model_id)
        assert len(measurements) == 1
        assert measurements[0] >= 10  # Should be at least 10ms

    def test_get_latency_measurements_with_time_window(self):
        """Test getting latency measurements within specific time window"""
        collector = MetricsService()
        model_id = uuid4()

        # Record multiple measurements
        for _ in range(3):
            start_time = time.perf_counter()
            time.sleep(0.01)
            collector.record_inference_end(model_id=model_id, start_time=start_time)

        # Get measurements from last 60 seconds
        measurements = collector.get_latency_measurements(model_id=model_id, time_window=60)
        assert len(measurements) == 3

        # Get measurements from last 0 seconds (should be empty)
        measurements = collector.get_latency_measurements(model_id=model_id, time_window=0)
        assert len(measurements) == 0

    def test_get_latency_measurements_different_models(self):
        """Test that measurements are isolated by model ID"""
        collector = MetricsService()
        model_id_1 = uuid4()
        model_id_2 = uuid4()

        # Record measurements for both models
        start_time = time.perf_counter()
        collector.record_inference_end(model_id=model_id_1, start_time=start_time)
        collector.record_inference_end(model_id=model_id_2, start_time=start_time)

        time.sleep(0.01)
        start_time = time.perf_counter()
        collector.record_inference_end(model_id=model_id_2, start_time=start_time)

        # Each model should only see its own measurements
        measurements_1 = collector.get_latency_measurements(model_id=model_id_1)
        measurements_2 = collector.get_latency_measurements(model_id=model_id_2)

        assert len(measurements_1) == 1
        assert len(measurements_2) == 2

    def test_time_window_measurements(self):
        """Test that old measurements are filtered out by time window"""
        collector = MetricsService()
        model_id = uuid4()

        # First add a current measurement
        start_time = time.perf_counter()
        collector.record_inference_end(model_id=model_id, start_time=start_time)

        # Then add an "old" measurement by mocking datetime
        with patch("app.services.metrics_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now(UTC) - timedelta(seconds=90)
            collector.record_inference_end(model_id, start_time)

        # Only the first measurement should be returned with default 60s window
        measurements = collector.get_latency_measurements(model_id=model_id)
        assert len(measurements) == 1

        # Both should be returned with larger time window
        measurements = collector.get_latency_measurements(model_id=model_id, time_window=120)
        assert len(measurements) == 2

    def test_latency_measurement_timestamp(self):
        """Test that measurements are recorded with correct timestamps"""
        collector = MetricsService()
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        fixed_timestamp = fixed_time.timestamp()
        model_id = uuid4()
        start_time = collector.record_inference_start()

        with patch("app.services.metrics_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            collector.record_inference_end(model_id, start_time)

        # Get all entries and check for our timestamp
        with collector._lock:
            # Check the array for our timestamp
            found_matching_record = False
            for i in range(MAX_MEASUREMENTS):
                if (
                    collector._array[i]["model_id"] == str(model_id)
                    and abs(collector._array[i]["timestamp"] - fixed_timestamp) < 0.001
                ):
                    found_matching_record = True
                    break
            assert found_matching_record

    def test_empty_measurements(self):
        """Test behavior when no measurements exist"""
        collector = MetricsService()
        collector.reset()
        model_id = uuid4()

        measurements = collector.get_latency_measurements(model_id=model_id)
        assert measurements == []

    def test_latency_calculation_accuracy(self):
        """Test that latency calculation is reasonably accurate"""
        collector = MetricsService()
        model_id = uuid4()

        start_time = time.perf_counter()
        time.sleep(0.01)  # 10ms
        collector.record_inference_end(model_id=model_id, start_time=start_time)
        measurements = collector.get_latency_measurements(model_id=model_id)
        latency_ms = measurements[0]

        # Should be approximately 10ms (with some tolerance for timing precision)
        assert 10.0 <= latency_ms <= 12.0, f"Latency was {latency_ms} ms"

    def test_max_age_initialization_and_update(self):
        """Test that max_age_seconds is properly initialized and updated"""
        collector = MetricsService()
        assert collector._max_age_seconds == 60

        collector.update_max_age(120)
        assert collector._max_age_seconds == 120

    def test_shared_memory_usage(self):
        """Test that the shared memory is properly used"""
        collector = MetricsService()
        model_id = uuid4()
        start_time = time.perf_counter()
        collector.record_inference_end(model_id, start_time)

        # Create a new collector which should access the same shared memory
        collector2 = MetricsService()
        measurements = collector2.get_latency_measurements(model_id)

        assert len(measurements) == 1

    def test_circular_buffer(self):
        """Test that the circular buffer works correctly when exceeding MAX_MEASUREMENTS"""
        collector = MetricsService()
        model_id = uuid4()

        # Fill more than MAX_MEASUREMENTS entries
        head_before = collector._head
        for i in range(MAX_MEASUREMENTS + 10):
            collector.record_inference_end(model_id, time.perf_counter())

        # Check that the head has wrapped around correctly
        assert collector._head == head_before + MAX_MEASUREMENTS + 10

        # Check that we can still retrieve measurements
        measurements = collector.get_latency_measurements(model_id)
        assert len(measurements) == MAX_MEASUREMENTS

    def test_get_throughput_measurements_empty(self):
        """Test getting throughput measurements when no data exists"""
        collector = MetricsService()
        model_id = uuid4()

        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 0
        assert throughput_data == []

    def test_get_throughput_measurements_single_inference(self):
        """Test throughput measurements with a single inference"""
        collector = MetricsService()
        model_id = uuid4()

        # Record a single inference
        start_time = time.perf_counter()
        collector.record_inference_end(model_id=model_id, start_time=start_time)

        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 1
        assert len(throughput_data) == 1
        assert throughput_data[0][1] == 1  # One inference in that second

    def test_get_throughput_measurements_multiple_inferences_same_second(self):
        """Test throughput measurements with multiple inferences in the same second"""
        collector = MetricsService()
        model_id = uuid4()

        # Record multiple inferences rapidly (within same second)
        start_time = time.perf_counter()
        for _ in range(5):
            collector.record_inference_end(model_id=model_id, start_time=start_time)

        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 5
        assert len(throughput_data) == 1  # All in same second
        assert throughput_data[0][1] == 5  # Five inferences in that second

    def test_get_throughput_measurements_different_seconds(self):
        """Test throughput measurements across different seconds"""
        collector = MetricsService()
        model_id = uuid4()

        current_time = datetime.now(UTC)

        # Mock different timestamps to simulate inferences in different seconds
        with patch("app.services.metrics_service.datetime") as mock_datetime:
            # First second (recent) with 2 inferences
            mock_datetime.now.return_value = current_time
            collector.record_inference_end(model_id, time.perf_counter())
            collector.record_inference_end(model_id, time.perf_counter())

            # Second second (1 second ago) with 1 inference
            mock_datetime.now.return_value = current_time - timedelta(seconds=1)
            collector.record_inference_end(model_id, time.perf_counter())

            # Third second (2 seconds ago) with 3 inferences
            mock_datetime.now.return_value = current_time - timedelta(seconds=2)
            collector.record_inference_end(model_id, time.perf_counter())
            collector.record_inference_end(model_id, time.perf_counter())
            collector.record_inference_end(model_id, time.perf_counter())

            # Reset mock to current time for the measurement call
            mock_datetime.now.return_value = current_time

        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 6
        assert len(throughput_data) == 3  # Three different seconds

        # Sort to ensure consistent ordering
        throughput_data.sort()

        # Check counts per second
        assert throughput_data[0][1] == 3  # Third second (oldest): 3 inferences
        assert throughput_data[1][1] == 1  # Second second: 1 inference
        assert throughput_data[2][1] == 2  # First second (newest): 2 inferences

    def test_get_throughput_measurements_time_window_filter(self):
        """Test that throughput measurements respect the time window"""
        collector = MetricsService()
        model_id = uuid4()

        current_time = datetime.now(UTC)

        with patch("app.services.metrics_service.datetime") as mock_datetime:
            # Record inference within time window
            mock_datetime.now.return_value = current_time
            collector.record_inference_end(model_id, time.perf_counter())

            # Record inference outside time window (91 seconds ago)
            mock_datetime.now.return_value = current_time - timedelta(seconds=91)
            collector.record_inference_end(model_id, time.perf_counter())

            # Reset to current time for measurement calls
            mock_datetime.now.return_value = current_time

        # Get measurements for last 60 seconds
        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 1  # Only the recent inference
        assert len(throughput_data) == 1

        # Get measurements for last 120 seconds (should include both)
        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=120)

        assert total_inferences == 2  # Both inferences
        assert len(throughput_data) == 2

    def test_get_throughput_measurements_different_models(self):
        """Test that throughput measurements are isolated by model ID"""
        collector = MetricsService()
        model_id_1 = uuid4()
        model_id_2 = uuid4()

        # Record inferences for both models
        start_time = time.perf_counter()
        collector.record_inference_end(model_id_1, start_time)
        collector.record_inference_end(model_id_1, start_time)
        collector.record_inference_end(model_id_2, start_time)

        # Check model 1 throughput
        total_inferences_1, throughput_data_1 = collector.get_throughput_measurements(
            model_id=model_id_1, time_window=60
        )
        assert total_inferences_1 == 2
        assert len(throughput_data_1) == 1
        assert throughput_data_1[0][1] == 2

        # Check model 2 throughput
        total_inferences_2, throughput_data_2 = collector.get_throughput_measurements(
            model_id=model_id_2, time_window=60
        )
        assert total_inferences_2 == 1
        assert len(throughput_data_2) == 1
        assert throughput_data_2[0][1] == 1

    def test_get_throughput_measurements_sorted_output(self):
        """Test that throughput data is sorted by timestamp"""
        collector = MetricsService()
        model_id = uuid4()

        current_time = datetime.now(UTC)

        with patch("app.services.metrics_service.datetime") as mock_datetime:
            # Record inferences in reverse chronological order
            mock_datetime.now.return_value = current_time - timedelta(seconds=2)
            collector.record_inference_end(model_id, time.perf_counter())

            mock_datetime.now.return_value = current_time
            collector.record_inference_end(model_id, time.perf_counter())

            mock_datetime.now.return_value = current_time - timedelta(seconds=1)
            collector.record_inference_end(model_id, time.perf_counter())

            # Reset to current time for measurement call
            mock_datetime.now.return_value = current_time

        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)

        assert total_inferences == 3
        assert len(throughput_data) == 3

        # Verify timestamps are sorted in ascending order
        timestamps = [data[0] for data in throughput_data]
        assert timestamps == sorted(timestamps)

    def test_reset_clears_data(self):
        """Test that reset clears all throughput measurement data"""
        collector = MetricsService()
        model_id = uuid4()

        # Record some inferences
        start_time = time.perf_counter()
        collector.record_inference_end(model_id, start_time)
        collector.record_inference_end(model_id, start_time)

        # Verify data exists
        latency_measurements = collector.get_latency_measurements(model_id=model_id, time_window=60)
        assert len(latency_measurements) == 2
        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)
        assert total_inferences == 2
        assert len(throughput_data) == 1

        # Reset and verify data is cleared
        collector.reset()
        latency_measurements = collector.get_latency_measurements(model_id=model_id, time_window=60)
        assert len(latency_measurements) == 0
        total_inferences, throughput_data = collector.get_throughput_measurements(model_id=model_id, time_window=60)
        assert total_inferences == 0
        assert len(throughput_data) == 0
