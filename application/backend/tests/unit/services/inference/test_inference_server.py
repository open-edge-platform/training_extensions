# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, Mock, patch
from uuid import uuid4

import numpy as np
import pytest
from model_api.models import Model

from app.models import BatchInferenceInput, DatasetItemAnnotation, Label
from app.models.model_revision import ModelFormat, ModelPrecision, ModelVariant
from app.models.system import DeviceInfo, DeviceType
from app.services import ModelService
from app.services.inference import InferenceModel, InferenceServer, InferenceState, InferenceStatus
from app.services.inference.model_loader import LoadedModelHandle


class TestInferenceServer:
    def test_set_inference_model(self, tmp_path) -> None:
        project_id = uuid4()
        model_id = uuid4()
        model_variant_id = uuid4()
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        model_variant = Mock(
            spec=ModelVariant, id=model_variant_id, format=ModelFormat.OPENVINO, precision=ModelPrecision.FP16
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))

        model = Mock(spec=Model)
        model_handle = LoadedModelHandle(
            model_id=model_id, variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
        )

        with (
            patch.object(
                target=ModelService, attribute="get_model_variants", return_value=[model_variant]
            ) as mock_get_model_variants,
            patch.object(
                target=ModelService,
                attribute="get_model_binary_files",
                return_value=(True, (tmp_path / "model.xml", tmp_path / "model.bin")),
            ) as mock_get_model_binary_files,
            patch("app.services.inference.model_loader.ModelLoader.load", return_value=model_handle) as mock_load_model,
        ):
            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=model_id, device=device, ttl=60
            )

            assert model_loaded
            assert (
                inference_server._loaded_model is not None
                and inference_server._loaded_model.model_id == model_id
                and inference_server._loaded_model.model == model
                and inference_server._loaded_model.device == device
            )

            mock_get_model_variants.assert_called_once_with(project_id=project_id, model_id=model_id)
            mock_get_model_binary_files.assert_called_once_with(
                project_id=project_id, model_id=model_id, model_variant_id=model_variant_id
            )
            mock_load_model.assert_called_once_with(
                model_id=model_id,
                variant_id=model_variant_id,
                model_xml_path=tmp_path / "model.xml",
                device=device,
            )

    def test_set_inference_same_model_already_loaded(self, tmp_path) -> None:
        """Tests that inference server doesn't try to load the model again if it is already loaded."""
        project_id = uuid4()
        model_id = uuid4()
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        model = Mock(spec=Model)
        model_handle = LoadedModelHandle(
            model_id=model_id, variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = model_handle

        with patch(
            "app.services.inference.model_loader.ModelLoader.load", return_value=model_handle
        ) as mock_load_model:
            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=model_id, device=device, ttl=60
            )

            mock_load_model.assert_not_called()
            assert not model_loaded
            assert inference_server._loaded_model.model_id == model_id
            assert inference_server._loaded_model.model == model
            assert inference_server._loaded_model.device == device

    def test_set_inference_different_model_already_loaded(self, tmp_path) -> None:
        """Tests that inference server first unloads the old model before loading the new one."""
        project_id = uuid4()
        model_id, new_model_id = uuid4(), uuid4()
        new_model_variant_id = uuid4()
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        model_variant = Mock(
            spec=ModelVariant, id=new_model_variant_id, format=ModelFormat.OPENVINO, precision=ModelPrecision.FP16
        )

        model = Mock(spec=Model)
        model_handle = LoadedModelHandle(
            model_id=model_id, variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
        )
        new_model_handle = LoadedModelHandle(
            model_id=new_model_id, variant_id=new_model_variant_id, model=model, device=device, loaded_at=datetime.now()
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = model_handle

        with (
            patch.object(
                target=ModelService, attribute="get_model_variants", return_value=[model_variant]
            ) as mock_get_model_variants,
            patch.object(
                target=ModelService,
                attribute="get_model_binary_files",
                return_value=(True, (tmp_path / "model.xml", tmp_path / "model.bin")),
            ) as mock_get_model_binary_files,
            patch(
                "app.services.inference.model_loader.ModelLoader.load", return_value=new_model_handle
            ) as mock_load_model,
            patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload_model,
        ):
            model_loaded = inference_server.set_inference_model(
                project_id=project_id, model_id=new_model_id, device=device, ttl=60
            )

            mock_get_model_variants.assert_called_once_with(project_id=project_id, model_id=new_model_id)
            mock_get_model_binary_files.assert_called_once_with(
                project_id=project_id, model_id=new_model_id, model_variant_id=new_model_variant_id
            )
            mock_unload_model.assert_called_once_with(model_handle)
            mock_load_model.assert_called_once_with(
                model_id=new_model_id,
                variant_id=new_model_variant_id,
                model_xml_path=tmp_path / "model.xml",
                device=device,
            )
            assert model_loaded
            assert inference_server._loaded_model.model_id == new_model_id
            assert inference_server._loaded_model.model == model
            assert inference_server._loaded_model.device == new_model_handle.device

    def test_get_status_idle(self, tmp_path) -> None:
        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = None

        status = inference_server.get_status()

        assert status == InferenceState(status=InferenceStatus.IDLE)

    def test_get_status_active(self, tmp_path) -> None:
        model_id = uuid4()
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        model = Mock(spec=Model)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = LoadedModelHandle(
            model_id=model_id, variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
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
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        model = Mock(spec=Model)
        model_handle = LoadedModelHandle(
            model_id=uuid4(), variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
        )

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = model_handle

        with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload_model:
            inference_server.stop()

            mock_unload_model.assert_called_once_with(model_handle)
            assert inference_server._loaded_model is None

    def test_infer_batch_not_loaded(self, tmp_path) -> None:
        label = Mock(spec=Label)
        input = Mock(spec=BatchInferenceInput)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = None

        with pytest.raises(RuntimeError):
            inference_server.infer_batch(labels=[label], inputs=[input])

    def test_infer_batch(self, tmp_path) -> None:
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        media_id = uuid4()

        model = Mock(spec=Model)
        inference_result = Mock()
        model.infer_batch.return_value = [inference_result]

        label = Mock(spec=Label)
        raw_uint8 = np.full((20, 20, 3), 128, dtype=np.uint8)
        input = BatchInferenceInput(media_id=media_id, frame_index=15, data=raw_uint8)

        annotation = Mock(spec=DatasetItemAnnotation)

        inference_server = InferenceServer(data_dir=Path(tmp_path))
        inference_server._loaded_model = LoadedModelHandle(
            model_id=uuid4(), variant_id=uuid4(), model=model, device=device, loaded_at=datetime.now()
        )

        with patch("app.services.inference.inference_server.convert_prediction") as mock_convert_prediction:
            mock_convert_prediction.return_value = [annotation]
            result = inference_server.infer_batch(labels=[label], inputs=[input])

        (passed_batch,) = model.infer_batch.call_args.args
        assert len(passed_batch) == 1
        np.testing.assert_array_equal(passed_batch[0], raw_uint8)
        assert passed_batch[0].dtype == np.uint8

        assert result == {(media_id, 15): [annotation]}
