# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch
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
from app.models.model_revision import ModelFormat, ModelPrecision, ModelVariant
from app.services import ModelService
from app.services.inference import InferenceModel, InferenceServer, InferenceState, InferenceStatus
from app.services.inference.inference_server import _LoadedModel


class TestInferenceServer:
    def test_set_inference_model(self, tmp_path) -> None:
        project_id = uuid4()
        model_id = uuid4()
        model_variant_id = uuid4()
        device = "AUTO"

        model_variant = MagicMock(
            spec=ModelVariant, id=model_variant_id, format=ModelFormat.OPENVINO, precision=ModelPrecision.FP16
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))

        model = MagicMock(spec=Model)

        with (
            patch("model_api.models.Model.create_model", return_value=model) as mock_create_model,
            patch.object(
                target=ModelService, attribute="get_model_variants", return_value=[model_variant]
            ) as mock_get_model_variants,
            patch.object(
                target=ModelService,
                attribute="get_model_binary_files",
                return_value=(True, (tmp_path / "model.xml", tmp_path / "model.bin")),
            ) as mock_get_model_binary_files,
            patch("app.services.inference.inference_server.needs_float32_input", return_value=False) as mock_needs_fp32,
            patch("app.services.inference.inference_server.create_core") as mock_create_core,
            patch("app.services.inference.inference_server.OpenvinoAdapter") as mock_ov_adapter,
        ):
            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=model_id, device=device, ttl=60
            )

            assert model_loaded
            assert (
                inference_server._loaded_model is not None
                and inference_server._loaded_model.id == model_id
                and inference_server._loaded_model.model == model
                and inference_server._loaded_model.device == device
                and inference_server._loaded_model.float32_input is False
            )

            mock_get_model_variants.assert_called_once_with(project_id=project_id, model_id=model_id)
            mock_get_model_binary_files.assert_called_once_with(
                project_id=project_id, model_id=model_id, model_variant_id=model_variant_id
            )
            mock_needs_fp32.assert_called_once_with(tmp_path / "model.xml")
            mock_ov_adapter.assert_called_once_with(
                mock_create_core.return_value,
                str(tmp_path / "model.xml"),
                device=device,
                max_num_requests=2,
            )
            mock_create_model.assert_called_once_with(mock_ov_adapter.return_value)

    def test_set_inference_model_already_loaded(self, tmp_path) -> None:
        project_id = uuid4()
        model_id = uuid4()
        device = "AUTO"

        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = _LoadedModel(
            id=model_id, model=model, device=device, load_timestamp=datetime.now(), float32_input=False
        )

        model_loaded = inference_server.set_inference_model(
            project_id=project_id, model_id=model_id, device=device, ttl=60
        )

        assert not model_loaded
        assert inference_server._loaded_model.id == model_id
        assert inference_server._loaded_model.model == model
        assert inference_server._loaded_model.device == device

    def test_set_inference_model_fp32(self, tmp_path) -> None:
        """IR with 0-1 normalisation scale → float32_input=True and _FP32OpenvinoAdapter is used."""
        project_id = uuid4()
        model_id = uuid4()
        model_variant_id = uuid4()
        device = "AUTO"

        model_variant = MagicMock(
            spec=ModelVariant, id=model_variant_id, format=ModelFormat.OPENVINO, precision=ModelPrecision.FP16
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        model = MagicMock(spec=Model)

        with (
            patch("model_api.models.Model.create_model", return_value=model),
            patch.object(ModelService, "get_model_variants", return_value=[model_variant]),
            patch.object(
                ModelService,
                "get_model_binary_files",
                return_value=(True, (tmp_path / "model.xml", tmp_path / "model.bin")),
            ),
            patch("app.services.inference.inference_server.needs_float32_input", return_value=True),
            patch("app.services.inference.inference_server.create_core"),
            patch("app.services.inference.inference_server._FP32OpenvinoAdapter") as mock_fp32_adapter,
        ):
            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=model_id, device=device, ttl=60
            )

        assert model_loaded
        assert inference_server._loaded_model is not None
        assert inference_server._loaded_model.float32_input is True
        mock_fp32_adapter.assert_called_once()

    def test_get_status_idle(self, tmp_path) -> None:
        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = None

        status = inference_server.get_status()

        assert status == InferenceState(status=InferenceStatus.IDLE)

    def test_get_status_active(self, tmp_path) -> None:
        model_id = uuid4()
        device = "AUTO"

        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = _LoadedModel(
            id=model_id, model=model, device=device, load_timestamp=datetime.now(), float32_input=False
        )

        status = inference_server.get_status()

        assert status == InferenceState(
            status=InferenceStatus.ACTIVE,
            model=InferenceModel(
                model_id=model_id,
                device=device,
                load_timestamp=ANY,
            ),
        )

    def test_stop(self, tmp_path) -> None:
        model = MagicMock(spec=Model)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = _LoadedModel(
            id=uuid4(), model=model, device="AUTO", load_timestamp=datetime.now(), float32_input=False
        )

        inference_server.stop()

        assert inference_server._loaded_model is None

    def test_infer_batch_not_loaded(self, tmp_path) -> None:
        label = MagicMock(spec=Label)
        input = MagicMock(spec=BatchInferenceInput)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
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

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = _LoadedModel(
            id=uuid4(), model=model, device="AUTO", load_timestamp=datetime.now(), float32_input=False
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

    def test_infer_batch_fp32_scales_input(self, tmp_path) -> None:
        """Images are scaled to float32/255 when the loaded model has float32_input=True."""
        media_id = uuid4()

        model = MagicMock(spec=Model)
        inference_result = MagicMock()
        model.infer_batch.return_value = [inference_result]

        label = MagicMock(spec=Label)
        raw_uint8 = np.full((20, 20, 3), 128, dtype=np.uint8)
        input_item = BatchInferenceInput(media_id=media_id, frame_index=0, data=raw_uint8)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = _LoadedModel(
            id=uuid4(), model=model, device="AUTO", load_timestamp=datetime.now(), float32_input=True
        )

        with patch("app.services.inference.inference_server.convert_prediction", return_value=[]):
            inference_server.infer_batch(labels=[label], inputs=[input_item])

        (passed_batch,) = model.infer_batch.call_args.args
        assert len(passed_batch) == 1
        np.testing.assert_array_almost_equal(passed_batch[0], raw_uint8.astype(np.float32) / 255.0)
        assert passed_batch[0].dtype == np.float32
