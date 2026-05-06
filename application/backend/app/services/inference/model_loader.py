# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from loguru import logger
from model_api.adapters import OpenvinoAdapter, create_core
from model_api.models import Model

from app.models.system import DeviceInfo

MODELAPI_NSTREAMS = os.getenv("MODELAPI_NSTREAMS", "2")


@dataclass(frozen=True)
class LoadedModelHandle:
    """Holds a loaded Model API model together with the identifiers used to load it."""

    model_id: UUID
    variant_id: UUID
    device: DeviceInfo
    model: Model
    loaded_at: datetime


class ModelLoader:
    """
    Responsible for loading and unloading OpenVINO models via Model API.

    This is a low-level utility shared by higher-level services (e.g. ActiveModelService, InferenceServer).
    It owns the lifecycle of the native OpenVINO objects and ensures they are properly released on unload.
    """

    @staticmethod
    def load(model_id: UUID, variant_id: UUID, model_xml_path: Path, device: DeviceInfo) -> LoadedModelHandle:
        """
        Load a model from an OpenVINO IR file onto the given device.

        Args:
            model_id: The identifier of the model to load.
            variant_id: The identifier of the variant to load the model from.
            model_xml_path: Path to the .xml model file (the .bin must be alongside it).
            device: The device to load the model on.

        Returns:
            A LoadedModelHandle containing the ready-to-use Model API model.
        """
        logger.debug("Loading model '{}' on device '{}'", model_xml_path, device)
        ie = create_core()
        adapter = OpenvinoAdapter(
            ie,
            str(model_xml_path),
            device=device.as_openvino,
            max_num_requests=int(MODELAPI_NSTREAMS),
        )
        model = Model.create_model(adapter)
        return LoadedModelHandle(
            model_id=model_id,
            variant_id=variant_id,
            device=device,
            model=model,
            loaded_at=datetime.now(),
        )

    @staticmethod
    def unload(handle: LoadedModelHandle) -> None:
        """
        Release all native OpenVINO resources held by the model.

        Explicitly deletes the compiled model and async queue from the adapter so that OpenVINO frees GPU/CPU memory
        immediately rather than waiting for the Python GC.

        Args:
            handle: The handle returned by a previous call to `load`.
        """
        logger.debug("Unloading model '{}'", handle.model_id)
        adapter = cast(OpenvinoAdapter, handle.model.inference_adapter)
        if hasattr(adapter, "async_queue"):
            del adapter.async_queue
        if hasattr(adapter, "compiled_model"):
            del adapter.compiled_model
