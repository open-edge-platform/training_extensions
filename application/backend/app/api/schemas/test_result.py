# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from app.models.sink import SinkTestResult
from app.models.source import SourceTestResult


class TestResult(BaseModel):
    reachable: bool
    latency_ms: float | None = None
    error: str | None = None

    @staticmethod
    def create(result: SinkTestResult | SourceTestResult) -> "TestResult":
        return TestResult(reachable=result.reachable, latency_ms=result.latency_ms, error=result.error)
