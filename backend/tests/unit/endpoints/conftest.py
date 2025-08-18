import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def fxt_client():
    return TestClient(app)
