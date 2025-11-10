# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

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


@pytest_asyncio.fixture
async def fxt_async_client() -> AsyncGenerator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
