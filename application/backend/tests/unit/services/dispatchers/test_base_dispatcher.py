# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pydantic_core
import pytest

from app.models.sink import BaseSinkConfig


def test_zero_rate_limit_raises_in_sink() -> None:
    with pytest.raises(pydantic_core.ValidationError):
        BaseSinkConfig(id=uuid4(), name="name", rate_limit=0, output_formats=[])  # type: ignore[bad_argument_type]
