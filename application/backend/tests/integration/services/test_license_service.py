# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from app.services.license_service import LicenseService


class TestLicenseService:
    """Tests for LicenseService"""

    @pytest.fixture
    def fxt_license_service(self, tmp_path: Path) -> LicenseService:
        return LicenseService(data_dir=tmp_path)

    def test_is_accepted_returns_false_initially(self, fxt_license_service: LicenseService) -> None:
        """License should not be accepted when no consent file exists."""
        assert not fxt_license_service.is_accepted()

    def test_accept_creates_consent_file(self, fxt_license_service: LicenseService, tmp_path: Path) -> None:
        """Accepting the license should create the consent marker file."""
        fxt_license_service.accept()

        consent_file = tmp_path / LicenseService.CONSENT_FILENAME
        assert consent_file.exists()

    def test_is_accepted_returns_true_after_accept(self, fxt_license_service: LicenseService) -> None:
        """License should be accepted after calling accept()."""
        fxt_license_service.accept()
        assert fxt_license_service.is_accepted()

    def test_accept_is_idempotent(self, fxt_license_service: LicenseService) -> None:
        """Calling accept() multiple times should not raise an error."""
        fxt_license_service.accept()
        fxt_license_service.accept()
        assert fxt_license_service.is_accepted()

    def test_is_accepted_returns_true_when_file_pre_exists(self, tmp_path: Path) -> None:
        """License should be accepted when the consent file already exists on disk."""
        (tmp_path / LicenseService.CONSENT_FILENAME).touch()

        service = LicenseService(data_dir=tmp_path)
        assert service.is_accepted()
