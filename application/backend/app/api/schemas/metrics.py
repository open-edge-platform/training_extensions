# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from app.models.metrics import InferenceMetrics, TimeWindow


class PipelineMetricsView(BaseModel):
    """Pipeline metrics response"""

    time_window: TimeWindow
    inference: InferenceMetrics

    model_config = {
        "json_schema_extra": {
            "example": {
                "time_window": {"start": "2025-08-25T10:00:00Z", "end": "2025-08-25T10:01:00Z", "time_window": 60},
                "inference": {
                    "latency": {"avg_ms": 15.1, "min_ms": 12.3, "max_ms": 30.4, "p95_ms": 25.4, "latest_ms": 15.6},
                    "throughput": {
                        "avg_requests_per_second": 66.7,
                        "total_requests": 4000,
                        "max_requests_per_second": 85.2,
                    },
                },
            }
        }
    }
