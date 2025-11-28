# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System API Endpoints"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_system_service
from app.schemas.system import DeviceInfo
from app.services import SystemService

router = APIRouter(prefix="/api")


@router.get("/system/devices")
async def get_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfo]:
    """Returns the list of available compute devices (CPU, Intel XPU, NVIDIA CUDA)."""
    return system_service.get_devices()


@router.get("/system/metrics/memory")
async def get_memory(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> dict:
    """Returns the used memory in MB and total available memory in MB."""
    used, total = system_service.get_memory_usage()
    return {"used": int(used), "total": int(total)}
