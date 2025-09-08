# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from app.services import MetricsService, PipelineService


@pytest.fixture
def fxt_pipeline_service(fxt_active_pipeline_service, fxt_metrics_service, fxt_condition) -> PipelineService:
    """Fixture to create a PipelineService instance with mocked dependencies."""
    return PipelineService(fxt_active_pipeline_service, fxt_metrics_service, fxt_condition)


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
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
        ):
            mock_get_pipeline_by_id.return_value = fxt_running_pipeline
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_running_pipeline.project_id, time_window=60)

        fxt_metrics_service.get_latency_measurements.assert_called_once_with(model_id=fxt_model.id, time_window=60)
        assert metrics.time_window.time_window == 60
        assert metrics.inference.latency.avg_ms == 20.0
        assert metrics.inference.latency.min_ms == 10.0
        assert metrics.inference.latency.max_ms == 30.0
        assert metrics.inference.latency.p95_ms == 29.0
        assert metrics.inference.latency.latest_ms == 30.0

    def test_get_pipeline_metrics_no_data(
        self, fxt_pipeline_service, fxt_metrics_service, fxt_running_pipeline, fxt_model
    ):
        """Test retrieving pipeline metrics when no latency data is available."""
        fxt_metrics_service.get_latency_measurements.return_value = []
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

    def test_get_pipeline_metrics_not_running(self, fxt_pipeline_service, fxt_default_pipeline):
        """Test retrieving metrics for pipeline that is not running raises error."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            pytest.raises(ValueError, match="Cannot get metrics for a pipeline that is not running."),
        ):
            mock_get_pipeline_by_id.return_value = fxt_default_pipeline
            fxt_pipeline_service.get_pipeline_metrics(fxt_default_pipeline.project_id)
