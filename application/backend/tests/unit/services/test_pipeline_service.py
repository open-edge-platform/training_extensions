# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from app.services import MetricsService, PipelineService
from app.services.data_collect import DataCollector


@pytest.fixture
def fxt_data_collector(fxt_active_pipeline_service) -> DataCollector:
    """Fixture to create a DataCollector instance with mocked dependencies."""
    return DataCollector(fxt_active_pipeline_service)


@pytest.fixture
def fxt_pipeline_service(
    fxt_active_pipeline_service, fxt_data_collector, fxt_metrics_service, fxt_condition
) -> PipelineService:
    """Fixture to create a PipelineService instance with mocked dependencies."""
    return PipelineService(fxt_active_pipeline_service, fxt_data_collector, fxt_metrics_service, fxt_condition)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


class TestPipelineServiceUnit:
    """Unit tests for PipelineService."""

    def test_get_pipeline_metrics_success(
        self, fxt_pipeline_service, fxt_metrics_service, fxt_running_pipeline, fxt_model
    ):
        """Test successfully retrieving pipeline metrics."""
        fxt_metrics_service.get_latency_measurements.return_value = [10.0, 15.0, 20.0, 25.0, 30.0]
        fxt_metrics_service.get_throughput_measurements.return_value = (
            100,
            [(1630000000.0, 20), (1630000001.0, 15), (1630000002.0, 25)],
        )

        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
        ):
            mock_get_pipeline_by_id.return_value = fxt_running_pipeline
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_running_pipeline.project_id, time_window=60)

        fxt_metrics_service.get_latency_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.latency.avg_ms == 20.0
        assert metrics.inference.latency.min_ms == 10.0
        assert metrics.inference.latency.max_ms == 30.0
        assert metrics.inference.latency.p95_ms == 29.0
        assert metrics.inference.latency.latest_ms == 30.0

        fxt_metrics_service.get_throughput_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.throughput.total_requests == 100
        assert metrics.inference.throughput.avg_requests_per_second == 100 / 60  # total / time_window
        assert metrics.inference.throughput.max_requests_per_second == 25  # max from throughput data

    def test_get_pipeline_metrics_no_data(
        self, fxt_pipeline_service, fxt_metrics_service, fxt_running_pipeline, fxt_model
    ):
        """Test retrieving pipeline metrics when no latency data is available."""
        fxt_metrics_service.get_latency_measurements.return_value = []
        fxt_metrics_service.get_throughput_measurements.return_value = (0, [])
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
        ):
            mock_get_pipeline_by_id.return_value = fxt_running_pipeline
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_running_pipeline.project_id, time_window=60)

        fxt_metrics_service.get_latency_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.time_window.time_window == 60
        assert metrics.inference.latency.avg_ms is None
        assert metrics.inference.latency.min_ms is None
        assert metrics.inference.latency.max_ms is None
        assert metrics.inference.latency.p95_ms is None
        assert metrics.inference.latency.latest_ms is None

        fxt_metrics_service.get_throughput_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.throughput.total_requests is None
        assert metrics.inference.throughput.avg_requests_per_second is None
        assert metrics.inference.throughput.max_requests_per_second is None

    def test_get_pipeline_metrics_not_running(self, fxt_pipeline_service, fxt_default_pipeline):
        """Test retrieving metrics for pipeline that is not running raises error."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            pytest.raises(ValueError, match="Cannot get metrics for a pipeline that is not running."),
        ):
            mock_get_pipeline_by_id.return_value = fxt_default_pipeline
            fxt_pipeline_service.get_pipeline_metrics(fxt_default_pipeline.project_id)

    def test_get_pipeline_metrics_single_measurement(
        self, fxt_pipeline_service, fxt_metrics_service, fxt_running_pipeline, fxt_model
    ):
        """Test pipeline metrics with a single throughput measurement."""
        fxt_metrics_service.get_latency_measurements.return_value = [15.0]
        fxt_metrics_service.get_throughput_measurements.return_value = (1, [(1630000000.0, 1)])

        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
        ):
            mock_get_pipeline_by_id.return_value = fxt_running_pipeline
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_running_pipeline.project_id, time_window=60)

        fxt_metrics_service.get_latency_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.latency.avg_ms == 15.0
        assert metrics.inference.latency.min_ms == 15.0
        assert metrics.inference.latency.max_ms == 15.0
        assert metrics.inference.latency.p95_ms == 15.0
        assert metrics.inference.latency.latest_ms == 15.0

        fxt_metrics_service.get_throughput_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.throughput.total_requests == 1
        assert metrics.inference.throughput.avg_requests_per_second == 1 / 60  # 1 inference over 60 seconds
        assert metrics.inference.throughput.max_requests_per_second == 1

    def test_get_pipeline_metrics_high_throughput_data(
        self, fxt_pipeline_service, fxt_metrics_service, fxt_running_pipeline, fxt_model
    ):
        """Test pipeline metrics with high throughput scenarios."""
        fxt_metrics_service.get_latency_measurements.return_value = [5.0, 6.0, 7.0, 8.0, 9.0]
        # Simulate high throughput: 1000 total inferences with varying per-second counts
        throughput_data = [
            (1630000000.0, 50),
            (1630000001.0, 75),
            (1630000002.0, 100),
            (1630000003.0, 80),
            (1630000004.0, 95),
        ]
        fxt_metrics_service.get_throughput_measurements.return_value = (1000, throughput_data)

        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
        ):
            mock_get_pipeline_by_id.return_value = fxt_running_pipeline
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_running_pipeline.project_id, time_window=60)

        fxt_metrics_service.get_latency_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.latency.avg_ms == 7.0
        assert metrics.inference.latency.min_ms == 5.0
        assert metrics.inference.latency.max_ms == 9.0
        assert metrics.inference.latency.p95_ms == 8.8
        assert metrics.inference.latency.latest_ms == 9.0

        fxt_metrics_service.get_throughput_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.inference.throughput.total_requests == 1000
        assert metrics.inference.throughput.avg_requests_per_second == 1000 / 60  # ~16.67
        assert metrics.inference.throughput.max_requests_per_second == 100  # highest count from data
