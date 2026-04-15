# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_license_service
from app.api.routers import system as system_router
from app.main import app
from app.services.license_service import LicenseService


@pytest.fixture
def fxt_license_service() -> Generator[Mock]:
    license_service = Mock(spec=LicenseService)
    app.dependency_overrides[get_license_service] = lambda: license_service
    yield license_service
    app.dependency_overrides.pop(get_license_service, None)


class TestSystemInfoEndpoint:
    @pytest.mark.parametrize(
        ("is_accepted", "expected_license_accepted"),
        [(False, False), (True, True)],
    )
    def test_returns_license_status_and_platform(
        self,
        fxt_license_service: Mock,
        fxt_client: TestClient,
        is_accepted: bool,
        expected_license_accepted: bool,
    ) -> None:
        """GET /api/system/info returns license_accepted and the current platform."""
        fxt_license_service.is_accepted.return_value = is_accepted

        with patch.object(system_router, "_get_platform", return_value="linux"):
            response = fxt_client.get("/api/system/info")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["license_accepted"] == expected_license_accepted
        assert data["platform"] == "linux"


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("win32", "windows"), ("darwin", "macos"), ("linux", "linux")],
)
def test_get_platform_maps_supported_platforms(platform_name: str, expected: str) -> None:
    with patch.object(system_router.sys, "platform", new=platform_name):
        assert system_router._get_platform() == expected
