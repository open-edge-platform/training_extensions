# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest

from app.services import ActivePipelineService, MetricsService


@pytest.fixture
def fxt_active_pipeline_service() -> MagicMock:
    return MagicMock(spec=ActivePipelineService)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
