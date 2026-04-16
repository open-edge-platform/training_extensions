# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_license_service
from app.main import app
from app.services.license_service import LicenseService


@pytest.fixture
def fxt_license_service() -> Generator[Mock]:
    license_service = Mock(spec=LicenseService)
    app.dependency_overrides[get_license_service] = lambda: license_service
    yield license_service
    app.dependency_overrides.pop(get_license_service, None)


class TestLicenseEndpoints:
    def test_accept_license_returns_accepted(self, fxt_license_service: Mock, fxt_client: TestClient) -> None:
        """POST /api/license/accept should return license_accepted=True on success."""
        fxt_license_service.accept.return_value = None

        response = fxt_client.post("/api/license/accept")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"license_accepted": True}
        fxt_license_service.accept.assert_called_once()

    def test_accept_license_returns_503_on_os_error(self, fxt_license_service: Mock, fxt_client: TestClient) -> None:
        """POST /api/license/accept should return 503 when the consent file cannot be written."""
        fxt_license_service.accept.side_effect = OSError("Permission denied")

        response = fxt_client.post("/api/license/accept")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Permission denied" in response.json()["detail"]
