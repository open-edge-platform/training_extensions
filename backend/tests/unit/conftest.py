# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest

from app.services import ActivePipelineService, MetricsService

# @pytest.fixture(scope="session", autouse=True)
# def set_multiprocessing_start_method():
#     # Set multiprocessing start method to 'fork' to ensure mocked objects and patches
#     # from the parent process are inherited by child processes. The default 'spawn'
#     # method creates isolated child processes that don't inherit mocked state.
#     mp.set_start_method("fork", force=True)


@pytest.fixture
def fxt_active_pipeline_service() -> MagicMock:
    return MagicMock(spec=ActivePipelineService)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
