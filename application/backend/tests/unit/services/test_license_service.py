# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from app.services.license_service import LicenseService


class TestLicenseService:
    """Test cases for LicenseService"""

    @pytest.fixture
    def fxt_data_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "data"

    @pytest.fixture
    def fxt_license_service(self, fxt_data_dir: Path) -> LicenseService:
        return LicenseService(data_dir=fxt_data_dir)

    def test_is_accepted_returns_false_initially(self, fxt_license_service: LicenseService) -> None:
        """License should not be accepted when no consent file exists."""
        assert fxt_license_service.is_accepted() is False

    def test_accept_creates_consent_file(self, fxt_license_service: LicenseService, fxt_data_dir: Path) -> None:
        """Accepting the license should create the consent marker file."""
        fxt_license_service.accept()

        consent_file = fxt_data_dir / LicenseService.CONSENT_FILENAME
        assert consent_file.exists()

    def test_is_accepted_returns_true_after_accept(self, fxt_license_service: LicenseService) -> None:
        """License should be accepted after calling accept()."""
        fxt_license_service.accept()
        assert fxt_license_service.is_accepted() is True

    def test_accept_creates_data_dir_if_missing(self, fxt_license_service: LicenseService, fxt_data_dir: Path) -> None:
        """accept() should create the data directory if it does not exist."""
        assert not fxt_data_dir.exists()
        fxt_license_service.accept()
        assert fxt_data_dir.exists()

    def test_accept_is_idempotent(self, fxt_license_service: LicenseService) -> None:
        """Calling accept() multiple times should not raise an error."""
        fxt_license_service.accept()
        fxt_license_service.accept()
        assert fxt_license_service.is_accepted() is True

    def test_is_accepted_returns_true_when_file_pre_exists(self, fxt_data_dir: Path) -> None:
        """License should be accepted when the consent file already exists on disk."""
        fxt_data_dir.mkdir(parents=True)
        (fxt_data_dir / LicenseService.CONSENT_FILENAME).touch()

        service = LicenseService(data_dir=fxt_data_dir)
        assert service.is_accepted() is True
