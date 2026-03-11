# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.services import DatasetRevisionService, DatasetService, LabelService, MediaService


@pytest.fixture
def fxt_db_session() -> Mock:
    """Create a mock database session."""
    return Mock()


@pytest.fixture
def fxt_db_session_factory(fxt_db_session: Mock):
    """Create a mock database session factory."""

    @contextmanager
    def session_factory():
        yield fxt_db_session

    return session_factory


@pytest.fixture
def fxt_staged_datasets_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "staged_datasets"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def fxt_dataset_service() -> Mock:
    """Mock DatasetService for testing."""
    return Mock(spec=DatasetService)


@pytest.fixture
def fxt_dataset_revision_service() -> Mock:
    """Mock DatasetRevisionService for testing."""
    return Mock(spec=DatasetRevisionService)


@pytest.fixture
def fxt_label_service() -> Mock:
    """Mock LabelService for testing."""
    return Mock(spec=LabelService)


@pytest.fixture
def fxt_media_service() -> Mock:
    """Mock MediaService for testing."""
    return Mock(spec=MediaService)
