# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class TestResult(BaseModel):
    reachable: bool
    latency_ms: float | None = None
    error: str | None = None
