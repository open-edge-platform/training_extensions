# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System schemas"""

from pydantic import BaseModel, Field

from app.models.system import DeviceType


class DeviceInfoView(BaseModel):
    """Device information schema"""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, or cuda)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"type": "cpu", "name": "CPU"},
                {"type": "xpu", "name": "Intel Arc B580", "memory": 12884901888, "index": 0},
                {"type": "cuda", "name": "NVIDIA GeForce RTX 4090", "memory": 25769803776, "index": 0},
            ]
        }
    }


class CameraInfoView(BaseModel):
    """Camera information schema"""

    index: int = Field(..., description="Camera device index")
    name: str = Field(..., description="Camera device name")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"index": 0, "name": "Integrated USB Camera"},
                {"index": 1, "name": "USB Camera"},
            ]
        }
    }
