# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import Mock

import pytest


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
