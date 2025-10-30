# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest

from app.services import MetricsService
from app.services.event.event_bus import EventBus


@pytest.fixture
def fxt_event_bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
