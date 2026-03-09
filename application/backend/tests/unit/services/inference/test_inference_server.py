# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from unittest.mock import ANY, MagicMock, call, patch
from uuid import uuid4

import numpy as np
import pytest
from model_api.models import Model

from app.models import (
    BatchInferenceInput,
    BatchInferenceMedia,
    BatchInferencePrediction,
    BatchInferenceResult,
    DatasetItemAnnotation,
    Label,
)
from app.services.inference import InferenceServer
from app.services.inference.inference_server import InferenceModel, InferenceState, InferenceStatus, LoadedModel


class TestInferenceServer:
    def test_set_inference_model(self, tmp_path) -> None:
        project_id = uuid4()
        model_id = uuid4()
        device = "AUTO"
        ttl = 60

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._get_model_file_path = MagicMock()
        inference_server._get_model_file_path.side_effect = [tmp_path / "model.xml", tmp_path / "model.bin"]

        model = MagicMock(spec=Model)

        with patch("model_api.models.Model.create_model") as mock_create_model:
            mock_create_model.return_value = model

            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=model_id, device=device, ttl=ttl
            )

            assert model_loaded
            assert (
                inference_server._loaded_model is not None
                and inference_server._loaded_model.id == model_id
                and inference_server._loaded_model.model == model
                and inference_server._loaded_model.device == device
                and inference_server._loaded_model.ttl == ttl
            )

            inference_server._get_model_file_path.assert_has_calls(
                [
                    call(project_id, model_id, "xml"),
                    call(project_id, model_id, "bin"),
                ]
            )
            mock_create_model.assert_called_once_with(model=str(tmp_path / "model.xml"), device=device, nstreams="2")

    def test_set_inference_model_already_loaded(self, tmp_path) -> None:
        project_id = uuid4()
        model_id = uuid4()
        device = "AUTO"
        ttl = 60

        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = LoadedModel(
            id=model_id, model=model, device=device, ttl=ttl, load_timestamp=datetime.now()
        )

        model_loaded = inference_server.set_inference_model(
            project_id=project_id, model_id=model_id, device=device, ttl=ttl
        )

        assert not model_loaded
        assert inference_server._loaded_model.id == model_id
        assert inference_server._loaded_model.model == model
        assert inference_server._loaded_model.device == device
        assert inference_server._loaded_model.ttl == ttl

    def test_get_status_idle(self, tmp_path) -> None:
        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = None

        status = inference_server.get_status()

        assert status == InferenceState(status=InferenceStatus.IDLE)

    def test_get_status_active(self, tmp_path) -> None:
        model_id = uuid4()
        device = "AUTO"
        ttl = 60

        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = LoadedModel(
            id=model_id, model=model, device=device, ttl=ttl, load_timestamp=datetime.now()
        )

        status = inference_server.get_status()

        assert status == InferenceState(
            status=InferenceStatus.ACTIVE,
            model=InferenceModel(
                model_id=model_id,
                device=device,
                ttl=ttl,
                load_timestamp=ANY,
                remaining_seconds=ANY,
            ),
        )

    def test_stop(self, tmp_path) -> None:
        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = LoadedModel(
            id=uuid4(), model=model, device="AUTO", ttl=60, load_timestamp=datetime.now()
        )

        inference_server.stop()

        assert inference_server._loaded_model is None

    def test_infer_batch_not_loaded(self, tmp_path) -> None:
        label = MagicMock(spec=Label)
        input = MagicMock(spec=BatchInferenceInput)

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = None

        with pytest.raises(RuntimeError):
            inference_server.infer_batch(labels=[label], inputs=[input])

    def test_infer_batch(self, tmp_path) -> None:
        media_id = uuid4()

        model = MagicMock(spec=Model)
        inference_result = MagicMock()
        model.infer_batch.return_value = [inference_result]

        label = MagicMock(spec=Label)
        input = BatchInferenceInput(media_id=media_id, frame_index=15, data=np.random.rand(20, 20, 3))

        annotation = MagicMock(spec=DatasetItemAnnotation)

        inference_server = InferenceServer(data_dir=tmp_path)
        inference_server._loaded_model = LoadedModel(
            id=uuid4(), model=model, device="AUTO", ttl=60, load_timestamp=datetime.now()
        )

        with patch("app.services.inference.inference_server.convert_prediction") as mock_convert_prediction:
            mock_convert_prediction.return_value = [annotation]
            result = inference_server.infer_batch(labels=[label], inputs=[input])

        assert result == BatchInferenceResult(
            predictions=[
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=media_id, frame_index=15), prediction=[annotation]
                )
            ]
        )
        model.infer_batch.assert_called_once_with([input.data])
        mock_convert_prediction.assert_called_once_with(
            labels=[label], frame_data=input.data, prediction=inference_result
        )
