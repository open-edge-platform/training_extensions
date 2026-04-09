# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""License consent service with file-based persistence."""

from pathlib import Path

from loguru import logger


class LicenseService:
    """Service to track third-party license consent using a file-based marker."""

    CONSENT_FILENAME = ".license_accepted"

    def __init__(self, data_dir: Path) -> None:
        self._consent_file = data_dir / self.CONSENT_FILENAME

    def is_accepted(self) -> bool:
        """Check whether the license has been accepted."""
        return self._consent_file.exists()

    def accept(self) -> None:
        """Record that the user accepted the license terms."""
        self._consent_file.touch()
        logger.info("License accepted — consent recorded at {}", self._consent_file)
