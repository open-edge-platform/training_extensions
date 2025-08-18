from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import status
from pydantic import ValidationError

from app.schemas import Pipeline, PipelineStatus
from app.services import ResourceInUseError, ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_pipeline() -> Pipeline:
    return Pipeline(
        id=uuid4(),
        name="test_pipeline",
        status=PipelineStatus.IDLE,
    )


class TestPipelineEndpoints:
    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_pipeline_by_id"),
            ("delete", "delete_pipeline_by_id"),
            ("patch", "update_pipeline"),
        ],
    )
    def test_pipeline_invalid_ids(self, http_method, service_method, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            getattr(mock_service.return_value, service_method).side_effect = None

            response = getattr(fxt_client, http_method)("/api/pipelines/invalid-id")

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            getattr(mock_service.return_value, service_method).assert_not_called()

    def test_list_pipeline_success(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.list_pipelines.return_value = [fxt_pipeline] * 2

            response = fxt_client.get("/api/pipelines")

            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()) == 2
            mock_service.return_value.list_pipelines.assert_called_once()

    def test_get_pipeline_success(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.get_pipeline_by_id.return_value = fxt_pipeline

            response = fxt_client.get(f"/api/pipelines/{str(fxt_pipeline.id)}")

            assert response.status_code == status.HTTP_200_OK
            mock_service.return_value.get_pipeline_by_id.assert_called_once_with(fxt_pipeline.id)

    def test_create_pipeline_success(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.create_pipeline.return_value = fxt_pipeline

            response = fxt_client.post("/api/pipelines", json={"name": "New Pipeline", "status": PipelineStatus.IDLE})

            assert response.status_code == status.HTTP_201_CREATED
            mock_service.return_value.create_pipeline.assert_called_once()

    def test_create_pipeline_invalid(self, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.create_pipeline.return_value = None

            response = fxt_client.post(
                "/api/pipelines", json={"name": "New Pipeline", "status": PipelineStatus.RUNNING}
            )

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            mock_service.return_value.create_pipeline.assert_not_called()

    def test_update_pipeline_success(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.return_value = fxt_pipeline

            response = fxt_client.patch(f"/api/pipelines/{str(fxt_pipeline.id)}", json={"name": "Updated Pipeline"})

            assert response.status_code == status.HTTP_200_OK
            mock_service.return_value.update_pipeline.assert_called_once_with(
                fxt_pipeline.id, {"name": "Updated Pipeline"}
            )

    def test_update_pipeline_status(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.return_value = None

            response = fxt_client.patch(f"/api/pipelines/{str(fxt_pipeline.id)}", json={"status": PipelineStatus.IDLE})

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            mock_service.return_value.update_pipeline.assert_not_called()

    def test_update_pipeline_not_found(self, fxt_pipeline, fxt_client):
        pipeline_id = str(fxt_pipeline.id)
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.side_effect = ResourceNotFoundError(
                ResourceType.PIPELINE, pipeline_id
            )

            response = fxt_client.patch(f"/api/pipelines/{pipeline_id}", json={"name": "Updated Pipeline"})

            assert response.status_code == status.HTTP_404_NOT_FOUND
            mock_service.return_value.update_pipeline.assert_called_once_with(
                fxt_pipeline.id, {"name": "Updated Pipeline"}
            )

    @pytest.mark.parametrize(
        "pipeline_op, pipeline_status",
        [
            ("enable", PipelineStatus.RUNNING),
            ("disable", PipelineStatus.IDLE),
        ],
    )
    def test_enable_pipeline(self, pipeline_op, pipeline_status, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.return_value = None

            response = fxt_client.post(f"/api/pipelines/{str(fxt_pipeline.id)}:{pipeline_op}")

            assert response.status_code == status.HTTP_204_NO_CONTENT
            mock_service.return_value.update_pipeline.assert_called_once_with(
                fxt_pipeline.id, {"status": pipeline_status}
            )

    @pytest.mark.parametrize("pipeline_op", ["enable", "disable"])
    def test_enable_pipeline_invalid_id(self, pipeline_op, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            response = fxt_client.post(f"/api/pipelines/invalid-id:{pipeline_op}")

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            mock_service.return_value.update_pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "pipeline_op, pipeline_status",
        [
            ("enable", PipelineStatus.RUNNING),
            ("disable", PipelineStatus.IDLE),
        ],
    )
    def test_enable_non_existent_pipeline(self, pipeline_op, pipeline_status, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.side_effect = ResourceNotFoundError(
                resource_type=ResourceType.PIPELINE,
                resource_id=str(fxt_pipeline.id),
            )

            response = fxt_client.post(f"/api/pipelines/{str(fxt_pipeline.id)}:{pipeline_op}")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            mock_service.return_value.update_pipeline.assert_called_once_with(
                fxt_pipeline.id, {"status": pipeline_status}
            )

    def test_cannot_enable_pipeline(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.update_pipeline.side_effect = ValidationError.from_exception_data(
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
            mock_service.return_value.update_pipeline.assert_called_once_with(
                fxt_pipeline.id, {"status": PipelineStatus.RUNNING}
            )

    def test_delete_pipeline_success(self, fxt_pipeline, fxt_client):
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.delete_pipeline_by_id.side_effect = None

            response = fxt_client.delete(f"/api/pipelines/{str(fxt_pipeline.id)}")

            assert response.status_code == status.HTTP_204_NO_CONTENT
            mock_service.return_value.delete_pipeline_by_id.assert_called_once_with(fxt_pipeline.id)

    def test_delete_pipeline_not_found(self, fxt_client):
        pipeline_id = str(uuid4())
        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            mock_service.return_value.delete_pipeline_by_id.side_effect = ResourceNotFoundError(
                ResourceType.PIPELINE, pipeline_id
            )

            response = fxt_client.delete(f"/api/pipelines/{pipeline_id}")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_pipeline_in_use(self, fxt_pipeline, fxt_client):
        pipeline_id = str(fxt_pipeline.id)

        with patch("app.api.endpoints.pipelines.PipelineService") as mock_service:
            err = ResourceInUseError(ResourceType.PIPELINE, pipeline_id)
            mock_service.return_value.delete_pipeline_by_id.side_effect = err

            response = fxt_client.delete(f"/api/pipelines/{pipeline_id}")

            assert response.status_code == status.HTTP_409_CONFLICT
            assert str(err) == response.json()["detail"]
