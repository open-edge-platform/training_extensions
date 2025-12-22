# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_system_service
from app.main import app
from app.schemas.system import DeviceInfo, DeviceType
from app.services import SystemService


@pytest.fixture
def fxt_client():
    return TestClient(app)


@pytest.fixture
def fxt_system_service() -> Mock:
    system_service = Mock(spec=SystemService)
    app.dependency_overrides[get_system_service] = lambda: system_service
    return system_service


class TestSystemEndpoints:
    def test_get_inference_devices_cpu_only(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/devices/inference with CPU only"""
        fxt_system_service.get_inference_devices.return_value = [
            DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
        ]

        response = fxt_client.get("/api/system/devices/inference")

        assert response.status_code == status.HTTP_200_OK
        devices = response.json()
        assert len(devices) == 1
        assert devices[0]["type"] == "cpu"
        assert devices[0]["name"] == "CPU"
        assert devices[0]["memory"] is None
        assert devices[0]["index"] is None

    def test_get_inference_devices_with_xpu(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/devices/inference with Intel XPU"""
        fxt_system_service.get_inference_devices.return_value = [
            DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
            DeviceInfo(type=DeviceType.XPU, name="Intel(R) Graphics [0x7d41]", memory=36022263808, index=0),
        ]

        response = fxt_client.get("/api/system/devices/inference")

        assert response.status_code == status.HTTP_200_OK
        devices = response.json()
        assert len(devices) == 2
        assert devices[0]["type"] == "cpu"
        assert devices[1]["type"] == "xpu"
        assert devices[1]["name"] == "Intel(R) Graphics [0x7d41]"
        assert devices[1]["memory"] == 36022263808
        assert devices[1]["index"] == 0

    def test_get_memory(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/metrics/memory"""
        fxt_system_service.get_memory_usage.return_value = (1024.5, 8192.0)

        response = fxt_client.get("/api/system/metrics/memory")

        assert response.status_code == status.HTTP_200_OK
        memory = response.json()
        assert memory["used"] == 1024
        assert memory["total"] == 8192
