# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status
from pydantic import ValidationError

from app.api.dependencies import get_pipeline_service
from app.main import app
from app.schemas import Pipeline, PipelineStatus
from app.schemas.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, TimeWindow
from app.services import (
    PipelineService,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
)


@pytest.fixture
def fxt_pipeline() -> Pipeline:
    return Pipeline(
        id=uuid4(),
        name="test_pipeline",
        status=PipelineStatus.IDLE,
    )


@pytest.fixture
def fxt_pipeline_service() -> MagicMock:
    pipeline_service = MagicMock(spec=PipelineService)
    app.dependency_overrides[get_pipeline_service] = lambda: pipeline_service
    return pipeline_service


class TestPipelineEndpoints:
    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_pipeline_by_id"),
            ("delete", "delete_pipeline_by_id"),
            ("patch", "update_pipeline"),
        ],
    )
    def test_pipeline_invalid_ids(self, http_method, service_method, fxt_pipeline_service, fxt_client):
        response = getattr(fxt_client, http_method)("/api/pipelines/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_pipeline_service, service_method).assert_not_called()

    def test_list_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.list_pipelines.return_value = [fxt_pipeline] * 2

        response = fxt_client.get("/api/pipelines")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_pipeline_service.list_pipelines.assert_called_once()

    def test_get_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.get_pipeline_by_id.return_value = fxt_pipeline

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}")

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.get_pipeline_by_id.assert_called_once_with(fxt_pipeline.id)

    def test_create_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.create_pipeline.return_value = fxt_pipeline

        response = fxt_client.post("/api/pipelines", json={"name": "New Pipeline", "status": PipelineStatus.IDLE})

        assert response.status_code == status.HTTP_201_CREATED
        fxt_pipeline_service.create_pipeline.assert_called_once()

    def test_create_pipeline_exists(self, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.create_pipeline.side_effect = ResourceAlreadyExistsError(
            resource_type=ResourceType.PIPELINE, resource_name="New Pipeline"
        )
        response = fxt_client.post("/api/pipelines", json={"name": "New Pipeline", "status": PipelineStatus.IDLE})

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_pipeline_service.create_pipeline.assert_called_once()

    def test_create_pipeline_invalid(self, fxt_pipeline_service, fxt_client):
        response = fxt_client.post("/api/pipelines", json={"name": "New Pipeline", "status": PipelineStatus.RUNNING})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_pipeline_service.create_pipeline.assert_not_called()

    def test_update_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.update_pipeline.return_value = fxt_pipeline

        response = fxt_client.patch(f"/api/pipelines/{str(fxt_pipeline.id)}", json={"name": "Updated Pipeline"})

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.update_pipeline.assert_called_once_with(fxt_pipeline.id, {"name": "Updated Pipeline"})

    def test_update_pipeline_status(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.patch(f"/api/pipelines/{str(fxt_pipeline.id)}", json={"status": PipelineStatus.IDLE})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_service.update_pipeline.assert_not_called()

    def test_update_pipeline_not_found(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        pipeline_id = str(fxt_pipeline.id)
        fxt_pipeline_service.update_pipeline.side_effect = ResourceNotFoundError(ResourceType.PIPELINE, pipeline_id)

        response = fxt_client.patch(f"/api/pipelines/{pipeline_id}", json={"name": "Updated Pipeline"})

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_pipeline_service.update_pipeline.assert_called_once_with(fxt_pipeline.id, {"name": "Updated Pipeline"})

    @pytest.mark.parametrize(
        "pipeline_op, pipeline_status",
        [
            ("enable", PipelineStatus.RUNNING),
            ("disable", PipelineStatus.IDLE),
        ],
    )
    def test_enable_pipeline(self, pipeline_op, pipeline_status, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.post(f"/api/pipelines/{str(fxt_pipeline.id)}:{pipeline_op}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_pipeline_service.update_pipeline.assert_called_once_with(fxt_pipeline.id, {"status": pipeline_status})

    @pytest.mark.parametrize("pipeline_op", ["enable", "disable"])
    def test_enable_pipeline_invalid_id(self, pipeline_op, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.post(f"/api/pipelines/invalid-id:{pipeline_op}")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_service.update_pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "pipeline_op, pipeline_status",
        [
            ("enable", PipelineStatus.RUNNING),
            ("disable", PipelineStatus.IDLE),
        ],
    )
    def test_enable_non_existent_pipeline(
        self, pipeline_op, pipeline_status, fxt_pipeline, fxt_pipeline_service, fxt_client
    ):
        fxt_pipeline_service.update_pipeline.side_effect = ResourceNotFoundError(
            resource_type=ResourceType.PIPELINE,
            resource_id=str(fxt_pipeline.id),
        )

        response = fxt_client.post(f"/api/pipelines/{str(fxt_pipeline.id)}:{pipeline_op}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_pipeline_service.update_pipeline.assert_called_once_with(fxt_pipeline.id, {"status": pipeline_status})

    def test_cannot_enable_pipeline(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.update_pipeline.side_effect = ValidationError.from_exception_data(
            "Pipeline",
            [
                {
                    "type": "missing",
                    "loc": ("name",),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )

        response = fxt_client.post(f"/api/pipelines/{str(fxt_pipeline.id)}:enable")

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_pipeline_service.update_pipeline.assert_called_once_with(
            fxt_pipeline.id, {"status": PipelineStatus.RUNNING}
        )

    def test_delete_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.delete(f"/api/pipelines/{str(fxt_pipeline.id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_pipeline_service.delete_pipeline_by_id.assert_called_once_with(fxt_pipeline.id)

    def test_delete_pipeline_not_found(self, fxt_pipeline_service, fxt_client):
        pipeline_id = str(uuid4())
        fxt_pipeline_service.delete_pipeline_by_id.side_effect = ResourceNotFoundError(
            ResourceType.PIPELINE, pipeline_id
        )

        response = fxt_client.delete(f"/api/pipelines/{pipeline_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_pipeline_in_use(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        pipeline_id = str(fxt_pipeline.id)
        err = ResourceInUseError(ResourceType.PIPELINE, pipeline_id)
        fxt_pipeline_service.delete_pipeline_by_id.side_effect = err

        response = fxt_client.delete(f"/api/pipelines/{pipeline_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]

    def test_get_pipeline_metrics_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        """Test successful retrieval of pipeline metrics with default time window."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=60),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=100.5, min_ms=50.0, max_ms=200.0, p95_ms=180.0, latest_ms=120.0)
            ),
        )
        fxt_pipeline_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics")

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.id, 60)

    def test_get_pipeline_metrics_invalid_pipeline_id(self, fxt_pipeline_service, fxt_client):
        """Test metrics endpoint with invalid pipeline ID format."""
        response = fxt_client.get("/api/pipelines/invalid-id/metrics")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_service.get_pipeline_metrics.assert_not_called()

    def test_get_pipeline_metrics_pipeline_not_found(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        """Test metrics endpoint when pipeline doesn't exist."""
        fxt_pipeline_service.get_pipeline_metrics.side_effect = ResourceNotFoundError(
            ResourceType.PIPELINE, str(fxt_pipeline.id)
        )

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.id, 60)

    def test_get_pipeline_metrics_pipeline_not_running(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        """Test metrics endpoint when pipeline is not in running state."""
        fxt_pipeline_service.get_pipeline_metrics.side_effect = ValueError(
            "Cannot get metrics for a pipeline that is not running."
        )

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot get metrics for a pipeline that is not running" in response.json()["detail"]
        fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.id, 60)

    @pytest.mark.parametrize("invalid_time_window", [0, -1, 3601, 7200])
    def test_get_pipeline_metrics_invalid_time_window(
        self, invalid_time_window, fxt_pipeline, fxt_pipeline_service, fxt_client
    ):
        """Test metrics endpoint with invalid time window values."""
        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics?time_window={invalid_time_window}")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Duration must be between 1 and 3600 seconds" in response.json()["detail"]
        fxt_pipeline_service.get_pipeline_metrics.assert_not_called()

    @pytest.mark.parametrize("valid_time_window", [1, 30, 300, 1800, 3600])
    def test_get_pipeline_metrics_valid_time_windows(
        self, valid_time_window, fxt_pipeline, fxt_pipeline_service, fxt_client
    ):
        """Test metrics endpoint with various valid time window values."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=valid_time_window),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=100.0, min_ms=50.0, max_ms=200.0, p95_ms=180.0, latest_ms=120.0)
            ),
        )
        fxt_pipeline_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics?time_window={valid_time_window}")

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.id, valid_time_window)

    def test_get_pipeline_metrics_no_data_available(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        """Test metrics endpoint when no latency data is available."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=60),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=None, min_ms=None, max_ms=None, p95_ms=None, latest_ms=None)
            ),
        )
        fxt_pipeline_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}/metrics")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["inference"]["latency"]["avg_ms"] is None
        assert response_data["inference"]["latency"]["min_ms"] is None
        assert response_data["inference"]["latency"]["max_ms"] is None
        assert response_data["inference"]["latency"]["p95_ms"] is None
        assert response_data["inference"]["latency"]["latest_ms"] is None
        fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.id, 60)
