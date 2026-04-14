# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_license_service
from app.api.routers import system as system_router
from app.main import app
from app.services.license_service import LicenseService


@pytest.fixture
def fxt_license_service() -> Generator[Mock, None, None]:
    license_service = Mock(spec=LicenseService)
    app.dependency_overrides[get_license_service] = lambda: license_service
    yield license_service
    app.dependency_overrides.pop(get_license_service, None)


class TestSystemInfoEndpoint:
    def test_returns_license_status_and_platform(
        self, fxt_license_service: Mock, fxt_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/system/info returns license_accepted and the current platform."""
        fxt_license_service.is_accepted.return_value = False
        monkeypatch.setattr(system_router, "_get_platform", lambda: "linux")

        response = fxt_client.get("/api/system/info")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data == {"license_accepted": False, "platform": "linux"}

        fxt_license_service.is_accepted.return_value = True
        response = fxt_client.get("/api/system/info")
        assert response.json()["license_accepted"] is True


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("win32", "windows"), ("darwin", "macos"), ("linux", "linux")],
)
def test_get_platform_maps_supported_platforms(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, expected: str
) -> None:
    monkeypatch.setattr(system_router.sys, "platform", platform_name)

    assert system_router._get_platform() == expected
