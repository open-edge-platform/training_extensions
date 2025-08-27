# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from otx.config import register_configs


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()
