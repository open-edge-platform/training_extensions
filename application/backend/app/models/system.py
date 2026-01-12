# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System schemas"""

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Enumeration of device types"""

    CPU = auto()
    XPU = auto()
    CUDA = auto()


class DeviceInfo(BaseModel):
    """Device information schema"""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, or cuda)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")


class CameraInfo(BaseModel):
    """Camera information schema"""

    index: int = Field(..., description="Camera device index")
    name: str = Field(..., description="Camera device name")
