# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def fxt_projects_dir(tmp_path: Path) -> Generator[Path]:
    """Set up a temporary data directory for tests."""
    projects_dir = Path(tmp_path / "projects")
    projects_dir.mkdir(parents=True, exist_ok=True)
    yield projects_dir
