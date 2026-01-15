# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System API Endpoints"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_system_service
from app.api.schemas.system import CameraInfoView, DeviceInfoView
from app.services import SystemService

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/devices/inference")
async def get_inference_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfoView]:
    """Returns the list of available compute devices (CPU, Intel XPU)."""
    inference_devices = system_service.get_inference_devices()
    return [DeviceInfoView.model_validate(device, from_attributes=True) for device in inference_devices]


@router.get("/devices/training")
async def get_training_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfoView]:
    """Returns the list of available training devices (CPU, Intel XPU, NVIDIA CUDA)."""
    training_devices = system_service.get_training_devices()
    return [DeviceInfoView.model_validate(device, from_attributes=True) for device in training_devices]


@router.get("/devices/camera")
async def get_camera_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[CameraInfoView]:
    """Returns the list of available camera devices."""
    camera_devices = system_service.get_camera_devices()
    return [CameraInfoView.model_validate(device, from_attributes=True) for device in camera_devices]


@router.get("/metrics/memory")
async def get_memory(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> dict:
    """Returns the used memory in MB and total available memory in MB."""
    used, total = system_service.get_memory_usage()
    return {"used": int(used), "total": int(total)}
