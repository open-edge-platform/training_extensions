# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from app.core.run import ExecutionContext
from app.services.training.base import PipelineContext


@pytest.fixture
def fxt_execution_ctx() -> Mock:
    """Mock ExecutionContext for testing."""
    return Mock(spec=ExecutionContext)


@pytest.fixture
def fxt_pipeline_ctx() -> PipelineContext:
    """Mock PipelineContext for testing."""
    return PipelineContext()
