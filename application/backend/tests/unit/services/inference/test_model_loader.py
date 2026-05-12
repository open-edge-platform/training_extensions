# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from model_api.adapters import OpenvinoAdapter
from model_api.models import Model

from app.models.system import DeviceInfo, DeviceType
from app.services.inference.model_loader import MODELAPI_NSTREAMS, LoadedModelHandle, ModelLoader


@pytest.fixture
def fxt_device_cpu() -> DeviceInfo:
    return DeviceInfo(type=DeviceType.CPU, name="cpu")


@pytest.fixture
def fxt_loaded_handle(fxt_device_cpu: DeviceInfo) -> LoadedModelHandle:
    fake_model = Mock(spec=Model)
    fake_model.inference_adapter = Mock(spec=OpenvinoAdapter)
    return LoadedModelHandle(
        model_id=uuid4(),
        variant_id=uuid4(),
        device=fxt_device_cpu,
        model=fake_model,
        loaded_at=datetime.now(),
    )


class TestModelLoader:
    def test_load_returns_handle_with_correct_ids(self, fxt_device_cpu: DeviceInfo, tmp_path: Path) -> None:
        """load() should return a LoadedModelHandle whose IDs match what was passed in."""
        model_id = uuid4()
        variant_id = uuid4()
        fake_model = Mock(spec=Model)
        fake_adapter = Mock(spec=OpenvinoAdapter)

        with (
            patch("app.services.inference.model_loader.create_core") as mock_create_core,
            patch("app.services.inference.model_loader.OpenvinoAdapter", return_value=fake_adapter) as mock_adapter_cls,
            patch(
                "app.services.inference.model_loader.Model.create_model", return_value=fake_model
            ) as mock_create_model,
        ):
            handle = ModelLoader.load(
                model_id=model_id,
                variant_id=variant_id,
                model_xml_path=tmp_path / "model.xml",
                device=fxt_device_cpu,
            )

            mock_adapter_cls.assert_called_once_with(
                mock_create_core.return_value,
                str(tmp_path / "model.xml"),
                device="CPU",
                max_num_requests=int(MODELAPI_NSTREAMS),
            )
            mock_create_model.assert_called_once_with(fake_adapter)

        assert handle.model_id == model_id
        assert handle.variant_id == variant_id
        assert handle.device == fxt_device_cpu
        assert handle.model is fake_model
        assert isinstance(handle.loaded_at, datetime)

    def test_unload_deletes_async_queue_and_compiled_model(self, fxt_loaded_handle: LoadedModelHandle) -> None:
        """unload() should delete both async_queue and compiled_model from the adapter."""
        adapter = cast(OpenvinoAdapter, fxt_loaded_handle.model.inference_adapter)
        adapter.async_queue = Mock()
        adapter.compiled_model = Mock()

        ModelLoader.unload(fxt_loaded_handle)

        assert not hasattr(adapter, "async_queue")
        assert not hasattr(adapter, "compiled_model")
