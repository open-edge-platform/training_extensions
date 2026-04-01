# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System schemas"""

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Enumeration of device types"""

    AUTO = auto()
    CPU = auto()
    XPU = auto()
    CUDA = auto()


class DeviceInfo(BaseModel):
    """Device information schema"""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, or cuda)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")

    @property
    def as_openvino(self) -> str:
        """
        Convert and validate DeviceInfo for OpenVINO device format

        Examples:
            DeviceInfo(type=DeviceType.CPU) -> "CPU"
            DeviceInfo(type=DeviceType.XPU, index=1) -> "GPU.1"

            DeviceInfo(type=DeviceType.GPU) -> raises ValueError()
        """
        if self.type == DeviceType.AUTO:
            return "AUTO"
        if self.type == DeviceType.CPU:
            return "CPU"
        if self.type == DeviceType.XPU:
            return f"GPU.{self.index}" if self.index is not None else "GPU"
        raise ValueError(f"Unsupported device type for OpenVINO: {self.type}")


class CameraInfo(BaseModel):
    """Camera information schema"""

    index: int = Field(..., description="Camera device index")
    name: str = Field(..., description="Camera device name")
