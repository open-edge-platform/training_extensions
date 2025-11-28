# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System schemas"""

from pydantic import BaseModel, Field


class DeviceInfo(BaseModel):
    """Device information schema"""

    type: str = Field(..., description="Device type (cpu, xpu, or cuda)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total device memory in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index (null for CPU)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"type": "cpu", "name": "CPU"},
                {"type": "xpu", "name": "Intel(R) Graphics [0x7d41]", "memory": 36022263808, "index": 0},
                {"type": "cuda", "name": "NVIDIA GeForce RTX 4090", "memory": 25769803776, "index": 0},
            ]
        }
    }
