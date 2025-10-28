# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def fxt_projects_dir() -> Generator[Path]:
    """Set up a temporary data directory for tests."""
    projects_dir = Path("data/projects")
    if not projects_dir.exists():
        projects_dir.mkdir(parents=True)
    yield projects_dir
    shutil.rmtree(projects_dir)
