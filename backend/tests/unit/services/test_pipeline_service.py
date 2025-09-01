# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.schemas import Model, Pipeline, PipelineStatus
from app.services import PipelineService


@pytest.fixture
def fxt_model() -> Model:
    """Fixture to provide a sample Model instance."""
    return Model(
        id=uuid4(),
        name="Test Model",
    )


@pytest.fixture(autouse=True)
def fxt_pipeline(fxt_model) -> Pipeline:
    """Fixture to provide a sample Pipeline instance."""
    return Pipeline(
        id=uuid4(),
        name="Test Pipeline",
        source_id=uuid4(),
        sink_id=uuid4(),
        model_id=fxt_model.id,
        status=PipelineStatus.RUNNING,
    )


@pytest.fixture
def fxt_pipeline_service(fxt_active_pipeline_service, fxt_condition) -> PipelineService:
    """Fixture to create a PipelineService instance with mocked dependencies."""
    return PipelineService(fxt_active_pipeline_service, fxt_condition)


class TestPipelineServiceUnit:
    """Unit tests for PipelineService."""

    def test_get_pipeline_metrics_success(self, fxt_pipeline_service, fxt_pipeline, fxt_model):
        """Test successfully retrieving pipeline metrics."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            patch("app.services.metrics_collector.MetricsCollector.get_latency_measurements") as mock_metrics_collector,
        ):
            mock_get_pipeline_by_id.return_value = fxt_pipeline
            mock_metrics_collector.return_value = [10.0, 15.0, 20.0, 25.0, 30.0]
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_pipeline.id, time_window=60)

        mock_metrics_collector.assert_called_once_with(fxt_model.id, 60)
        assert metrics.time_window.time_window == 60
        assert metrics.inference.latency.avg_ms == 20.0
        assert metrics.inference.latency.min_ms == 10.0
        assert metrics.inference.latency.max_ms == 30.0
        assert metrics.inference.latency.p95_ms == 29.0
        assert metrics.inference.latency.latest_ms == 30.0

    def test_get_pipeline_metrics_no_data(self, fxt_pipeline_service, fxt_pipeline, fxt_model):
        """Test retrieving pipeline metrics when no latency data is available."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            patch("app.services.metrics_collector.MetricsCollector.get_latency_measurements") as mock_metrics_collector,
        ):
            mock_get_pipeline_by_id.return_value = fxt_pipeline
            mock_metrics_collector.return_value = []
            metrics = fxt_pipeline_service.get_pipeline_metrics(fxt_pipeline.id, time_window=60)

        mock_metrics_collector.assert_called_once_with(fxt_model.id, 60)
        assert metrics.time_window.time_window == 60
        assert metrics.inference.latency.avg_ms is None
        assert metrics.inference.latency.min_ms is None
        assert metrics.inference.latency.max_ms is None
        assert metrics.inference.latency.p95_ms is None
        assert metrics.inference.latency.latest_ms is None

    def test_get_pipeline_metrics_model_not_found(self, fxt_pipeline_service, fxt_pipeline):
        """Test retrieving metrics for non-existent pipeline raises error."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            pytest.raises(
                ValueError, match="Cannot get metrics for a pipeline with no model, please check a model is assigned."
            ),
        ):
            mock_get_pipeline_by_id.return_value = fxt_pipeline.model_copy(update={"model_id": None})
            fxt_pipeline_service.get_pipeline_metrics(fxt_pipeline.id)

    def test_get_pipeline_metrics_not_running(self, fxt_pipeline_service, fxt_pipeline):
        """Test retrieving metrics for pipeline that is not running raises error."""
        with (
            patch("app.services.pipeline_service.PipelineService.get_pipeline_by_id") as mock_get_pipeline_by_id,
            pytest.raises(ValueError, match="Cannot get metrics for a pipeline that is not running."),
        ):
            mock_get_pipeline_by_id.return_value = fxt_pipeline.model_copy(update={"status": PipelineStatus.IDLE})
            fxt_pipeline_service.get_pipeline_metrics(fxt_pipeline.id)
