# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.api.dependencies import get_project
from app.main import app
from app.models import TaskType
from app.schemas import LabelView, ProjectView
from app.schemas.project import TaskView
from app.services import MetricsService
from app.services.event.event_bus import EventBus


@pytest.fixture
def fxt_get_project() -> Generator[ProjectView]:
    project = MagicMock(
        spec=ProjectView,
        id=uuid4(),
        task=TaskView(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=[
                LabelView(id=uuid4(), name="cat", color="#11AA22", hotkey="s"),
                LabelView(id=uuid4(), name="dog", color="#AA2233", hotkey="d"),
            ],
        ),
    )
    app.dependency_overrides[get_project] = lambda: project
    yield project
    del app.dependency_overrides[get_project]


@pytest.fixture
def fxt_event_bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
