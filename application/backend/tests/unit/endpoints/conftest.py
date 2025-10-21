# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_project_service
from app.main import app
from app.services import ProjectService


@pytest.fixture
def fxt_client():
    return TestClient(app)


@pytest.fixture
def fxt_project_service() -> Mock:
    project_service = Mock(spec=ProjectService)
    app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service
