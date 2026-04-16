# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from app.services.license_service import LicenseService

APP_VERSION = "1.0.0"


class TestLicenseService:
    """Tests for LicenseService"""

    @pytest.fixture
    def fxt_license_service(self, tmp_path: Path) -> LicenseService:
        return LicenseService(data_dir=tmp_path, app_version=APP_VERSION)

    def test_accept_creates_file_with_version_and_is_idempotent(
        self, fxt_license_service: LicenseService, tmp_path: Path
    ) -> None:
        """License starts not accepted; accept() creates a versioned marker file and is idempotent."""
        assert not fxt_license_service.is_accepted()

        fxt_license_service.accept()

        consent_file = tmp_path / LicenseService.CONSENT_FILENAME
        assert consent_file.exists()
        assert consent_file.read_text() == APP_VERSION
        assert fxt_license_service.is_accepted()

        assert LicenseService(data_dir=tmp_path, app_version=APP_VERSION).is_accepted()

        fxt_license_service.accept()
        assert fxt_license_service.is_accepted()

    def test_version_mismatch_requires_re_acceptance(self, tmp_path: Path) -> None:
        """License accepted for one version is not accepted for another; re-accepting overwrites."""
        old_service = LicenseService(data_dir=tmp_path, app_version="0.9.0")
        old_service.accept()

        new_service = LicenseService(data_dir=tmp_path, app_version=APP_VERSION)
        assert not new_service.is_accepted()

        new_service.accept()
        consent_file = tmp_path / LicenseService.CONSENT_FILENAME
        assert consent_file.read_text() == APP_VERSION
        assert new_service.is_accepted()

    def test_empty_marker_file_is_not_accepted(self, tmp_path: Path) -> None:
        """An empty consent file (e.g. legacy pre-version marker) is treated as not accepted."""
        (tmp_path / LicenseService.CONSENT_FILENAME).touch()

        service = LicenseService(data_dir=tmp_path, app_version=APP_VERSION)
        assert not service.is_accepted()
