# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_model_service
from app.api.schemas import ModelView
from app.main import app
from app.models import DatasetItemSubset, EvaluationResult, TrainingInfo, TrainingStatus
from app.models.model_revision import ModelFormat
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_model() -> ModelView:
    return ModelView(
        id=uuid4(),
        name="Object_Detection_YOLOX (id-short)",
        architecture="Object_Detection_YOLOX",
        training_info=TrainingInfo(status=TrainingStatus.NOT_STARTED, label_schema_revision={}, configuration={}),  # type: ignore
    )  # type: ignore


@pytest.fixture
def fxt_model_service() -> MagicMock:
    model_service = MagicMock(spec=ModelService)
    app.dependency_overrides[get_model_service] = lambda: model_service
    return model_service


class TestModelEndpoints:
    def test_list_model_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.list_models.return_value = [fxt_model] * 2
        fxt_model_service.get_model_variants.side_effect = [[], []]

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_model_service.list_models.assert_called_once_with(fxt_get_project.id)
        fxt_model_service.get_model_variants.assert_has_calls(
            [
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
            ]
        )

    def test_list_model_project_not_found(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.list_models.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.list_models.assert_called_once_with(fxt_get_project.id)

    def test_list_model_invalid_id(self, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.get("/api/projects/invalid-id/models")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.list_models.assert_not_called()

    def test_get_model_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.get_model.return_value = fxt_model
        fxt_model_service.get_model_variants.return_value = []
        fxt_model_service.get_evaluation_results.return_value = [
            EvaluationResult(
                model_revision_id=fxt_model.id,
                dataset_revision_id=uuid4(),
                subset=DatasetItemSubset.TRAINING,
                metrics={"accuracy": 0.95, "f1_score": 0.87},
            )
        ]

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)
        fxt_model_service.get_model_variants.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )
        fxt_model_service.get_evaluation_results.assert_called_once_with(model_id=fxt_model.id)

    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_model"),
            ("delete", "delete_model"),
        ],
    )
    def test_model_invalid_ids(self, http_method, service_method, fxt_get_project, fxt_model_service, fxt_client):
        response = getattr(fxt_client, http_method)(f"/api/projects/{fxt_get_project.id}/models/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_model_service, service_method).assert_not_called()

    def test_get_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_success(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)

    def test_delete_model_files_only_success(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}?files_only=true")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model_files.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_delete_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.delete_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_in_use(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        err = ResourceInUseError(ResourceType.MODEL, str(model_id))
        fxt_model_service.delete_model.side_effect = err

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    @pytest.mark.parametrize(
        "model_format, model_precision",
        [
            (ModelFormat.OPENVINO, "fp16"),
            (ModelFormat.ONNX, "fp16"),
            (ModelFormat.PYTORCH, "fp32"),
        ],
    )
    def test_download_model_binary_success(
        self, model_format, model_precision, fxt_get_project, fxt_model, fxt_model_service, fxt_client, tmp_path
    ):
        import zipfile
        from io import BytesIO

        # Create mock model files
        model_dir = tmp_path / "models" / str(fxt_model.id)
        model_dir.mkdir(parents=True)
        bin_content = b"binary model data"
        if model_format == ModelFormat.OPENVINO:
            xml_content = "<xml>model data</xml>"
            (model_dir / "model.xml").write_text(xml_content)
            (model_dir / "model.bin").write_bytes(bin_content)
        elif model_format == ModelFormat.ONNX:
            (model_dir / "model.onnx").write_bytes(bin_content)
        elif model_format == ModelFormat.PYTORCH:
            (model_dir / "model.ckpt").write_bytes(bin_content)

        fxt_model_service.get_model_binary_files.return_value = True, tuple(model_dir.glob("*"))

        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/binary?format={model_format.value}"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert (
            f"model-{fxt_model.id}-{model_format.value}-{model_precision}.zip"
            in response.headers["content-disposition"]
        )

        # Verify zip file contents
        zip_data = BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_file:
            if model_format == ModelFormat.OPENVINO:
                assert "model.xml" in zip_file.namelist()
                assert "model.bin" in zip_file.namelist()
                assert zip_file.read("model.xml").decode() == xml_content  # pyrefly: ignore[unbound-name]
                assert zip_file.read("model.bin") == bin_content
            elif model_format == ModelFormat.ONNX:
                assert "model.onnx" in zip_file.namelist()
                assert zip_file.read("model.onnx") == bin_content
            elif model_format == ModelFormat.PYTORCH:
                assert "model.ckpt" in zip_file.namelist()
                assert zip_file.read("model.ckpt") == bin_content

        fxt_model_service.get_model_binary_files.assert_called_once_with(
            project_id=fxt_get_project.id,
            model_id=fxt_model.id,
            format=model_format,
        )

    def test_download_model_binary_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model_binary_files.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model_binary_files.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=model_id, format=ModelFormat.OPENVINO
        )

    def test_download_model_binary_invalid_id(self, fxt_get_project, fxt_model_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/invalid-id/binary")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.get_model.assert_not_called()
