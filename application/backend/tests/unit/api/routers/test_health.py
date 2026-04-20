# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import status
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, fxt_client: TestClient) -> None:
        """GET /health should return status ok."""
        response = fxt_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok"}
